#!/usr/bin/env python3
"""Convert BFCL v3 evaluation rows with ground truth into no-think SFT data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def is_empty_choice(value: Any) -> bool:
    return value is None or value == ""


def choose_ground_truth_value(options: Any) -> Any:
    """Choose one explicit accepted value from BFCL's alternative-value lists."""
    if not isinstance(options, list):
        return canonicalize_nested_value(options)
    if not options:
        return ""
    nonempty = [value for value in options if not is_empty_choice(value)]
    selected = nonempty[0] if nonempty else options[0]
    return canonicalize_nested_value(selected)


def canonicalize_nested_value(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, options in value.items():
            selected = choose_ground_truth_value(options)
            if not is_empty_choice(selected):
                result[key] = selected
        return result
    if isinstance(value, tuple):
        return list(value)
    return value


def function_specs(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    functions = json.loads(row.get("functions") or "[]")
    return {
        str(function.get("name")): function
        for function in functions
        if isinstance(function, dict) and function.get("name")
    }


def format_python_value(value: Any) -> str:
    return repr(value)


def format_single_turn_calls(ground_truth: Any, specs: dict[str, dict[str, Any]]) -> str | None:
    if not isinstance(ground_truth, list) or not ground_truth:
        return None
    calls = []
    for call in ground_truth:
        if not isinstance(call, dict) or len(call) != 1:
            return None
        function_name, raw_arguments = next(iter(call.items()))
        if not isinstance(raw_arguments, dict):
            return None
        required = set(
            specs.get(function_name, {}).get("parameters", {}).get("required", [])
        )
        arguments = []
        for argument_name, options in raw_arguments.items():
            value = choose_ground_truth_value(options)
            if is_empty_choice(value) and argument_name not in required:
                continue
            arguments.append(f"{argument_name}={format_python_value(value)}")
        calls.append(f"{function_name}({', '.join(arguments)})")
    target = "[" + ", ".join(calls) + "]"
    return target


def format_multi_turn_calls(ground_truth: Any) -> str | None:
    if not isinstance(ground_truth, list) or not ground_truth:
        return None
    calls = [clean_text(call) for call in ground_truth if clean_text(call)]
    if not calls:
        return None
    target = "[" + ", ".join(calls) + "]"
    return target


def render_nothink(system: str, conversations: list[dict[str, str]]) -> str:
    chunks = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    for index in range(0, len(conversations), 2):
        chunks.append(
            f"<|im_start|>user\n{conversations[index]['value']}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            f"{conversations[index + 1]['value']}<|im_end|>\n"
        )
    return "".join(chunks)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def summarize_lengths(values: list[int]) -> dict[str, int | float]:
    return {
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
        "over_6000": sum(value > 6000 for value in values),
        "over_7000": sum(value > 7000 for value in values),
        "over_8192": sum(value > 8192 for value in values),
    }


def sample_key(sample: dict[str, Any]) -> str:
    payload = json.dumps(
        {"system": sample["system"], "conversations": sample["conversations"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = pq.read_table(args.parquet).to_pylist()
    samples = []
    removed = []
    seen = set()
    stats: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    lengths = []

    for row_index, row in enumerate(rows):
        stats["raw_rows"] += 1
        category = str(row.get("test_category", "unknown"))
        groups = json.loads(row.get("turns") or "[]")
        ground_truth = json.loads(row.get("ground_truth") or "{}")
        if not groups:
            removed.append({"id": row.get("id"), "reason": "empty_turns"})
            stats["removed_empty_turns"] += 1
            continue

        systems = [
            clean_text(message.get("content"))
            for group in groups
            for message in (group if isinstance(group, list) else [])
            if message.get("role") == "system" and clean_text(message.get("content"))
        ]
        if not systems:
            removed.append({"id": row.get("id"), "reason": "missing_system"})
            stats["removed_missing_system"] += 1
            continue
        system = systems[0]
        conversations: list[dict[str, str]] = []

        if row.get("multi_turn"):
            if not isinstance(ground_truth, list) or len(groups) != len(ground_truth):
                removed.append({"id": row.get("id"), "reason": "invalid_multi_turn_ground_truth"})
                stats["removed_invalid_multi_turn_ground_truth"] += 1
                continue
            pending_users = []
            for group, turn_ground_truth in zip(groups, ground_truth):
                users = [
                    clean_text(message.get("content"))
                    for message in (group if isinstance(group, list) else [])
                    if message.get("role") == "user" and clean_text(message.get("content"))
                ]
                user = "\n\n".join(users)
                target = format_multi_turn_calls(turn_ground_truth)
                if target is None:
                    stats["empty_multi_turn_targets"] += 1
                    if user:
                        pending_users.append(user)
                    continue
                merged_user = "\n\nFollow-up context:\n".join(
                    pending_users + ([user] if user else [])
                )
                pending_users.clear()
                if not merged_user:
                    stats["missing_user_for_target"] += 1
                    continue
                conversations.extend(
                    [
                        {"from": "human", "value": merged_user},
                        {"from": "gpt", "value": target},
                    ]
                )
        else:
            target = format_single_turn_calls(ground_truth, function_specs(row))
            if target is None:
                removed.append({"id": row.get("id"), "reason": "empty_ground_truth"})
                stats["removed_empty_ground_truth"] += 1
                continue
            users = [
                clean_text(message.get("content"))
                for message in groups[0]
                if message.get("role") == "user" and clean_text(message.get("content"))
            ]
            user = "\n\n".join(users)
            if not user:
                removed.append({"id": row.get("id"), "reason": "missing_user"})
                stats["removed_missing_user"] += 1
                continue
            conversations.extend(
                [
                    {"from": "human", "value": user},
                    {"from": "gpt", "value": target},
                ]
            )

        if not conversations:
            removed.append({"id": row.get("id"), "reason": "no_supervised_turns"})
            stats["removed_no_supervised_turns"] += 1
            continue

        sample = {
            "id": f"bfcl_v3:{row_index}:{row['id']}",
            "source": "bfcl_v3",
            "system": system,
            "conversations": conversations,
            "metadata": {
                "bfcl_id": row["id"],
                "category": category,
                "multi_turn": bool(row.get("multi_turn")),
                "supervised_turns": len(conversations) // 2,
                "empty_turn_targets_skipped": (
                    sum(not turn for turn in ground_truth) if row.get("multi_turn") else 0
                ),
                "canonical_ground_truth": not bool(row.get("multi_turn")),
            },
        }
        key = sample_key(sample)
        if key in seen:
            removed.append({"id": row.get("id"), "reason": "duplicate_sample"})
            stats["removed_duplicate_sample"] += 1
            continue
        seen.add(key)

        token_length = len(
            tokenizer.encode(
                render_nothink(sample["system"], sample["conversations"]),
                add_special_tokens=False,
            )
        )
        sample["metadata"]["raw_token_length"] = token_length
        lengths.append(token_length)
        categories[category] += 1
        samples.append(sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(args.output)

    removed_path = args.output.with_suffix(".removed.jsonl")
    removed_temp = removed_path.with_suffix(removed_path.suffix + ".tmp")
    with removed_temp.open("w", encoding="utf-8") as handle:
        for item in removed:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    removed_temp.replace(removed_path)

    summary = {
        "output": str(args.output),
        "removed_output": str(removed_path),
        "raw_rows": len(rows),
        "written_rows": len(samples),
        "removed_rows": len(removed),
        "single_turn_rows": sum(not sample["metadata"]["multi_turn"] for sample in samples),
        "multi_turn_rows": sum(sample["metadata"]["multi_turn"] for sample in samples),
        "supervised_turns": sum(sample["metadata"]["supervised_turns"] for sample in samples),
        "categories": dict(sorted(categories.items())),
        "counters": dict(sorted(stats.items())),
        "token_length": summarize_lengths(lengths),
    }
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
