#!/usr/bin/env python3
"""Measure BFCL v3 lengths using the Qwen3 no-think chat layout."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer


THRESHOLDS = (1600, 2048, 4096, 6000, 6144, 7000, 7168, 8192, 16384)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def format_target(value: Any) -> str | None:
    if not isinstance(value, list) or not value:
        return None
    if all(isinstance(item, str) for item in value):
        calls = [item.strip() for item in value if item.strip()]
        return "[" + ", ".join(calls) + "]" if calls else None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render(system: str, pairs: list[tuple[str, str]]) -> str:
    chunks = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    for user, assistant in pairs:
        chunks.append(
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            f"{assistant}<|im_end|>\n"
        )
    return "".join(chunks)


def summarize(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)

    def at(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]

    result: dict[str, int | float] = {
        "min": ordered[0],
        "p50": at(0.50),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }
    for threshold in THRESHOLDS:
        result[f"over_{threshold}"] = sum(value > threshold for value in ordered)
    return result


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = pq.read_table(args.parquet).to_pylist()
    entries = []
    category_counts: Counter[str] = Counter()
    category_trainable: Counter[str] = Counter()
    empty_target_rows: Counter[str] = Counter()
    empty_target_turns: Counter[str] = Counter()

    for row in rows:
        category = str(row.get("test_category", "unknown"))
        category_counts[category] += 1
        groups = json.loads(row["turns"])
        ground_truth = json.loads(row["ground_truth"])
        ground_truths = ground_truth if row.get("multi_turn") else [ground_truth]
        if not isinstance(groups, list) or len(groups) != len(ground_truths):
            raise ValueError(f"turn/ground-truth mismatch for {row['id']}")

        system = ""
        pairs: list[tuple[str, str]] = []
        pending_users: list[str] = []
        for group, turn_ground_truth in zip(groups, ground_truths):
            group = group if isinstance(group, list) else []
            if not system:
                systems = [str(message.get("content", "")).strip() for message in group if message.get("role") == "system"]
                system = next((value for value in systems if value), "")
            users = [str(message.get("content", "")).strip() for message in group if message.get("role") == "user"]
            user = "\n\n".join(value for value in users if value)
            target = format_target(turn_ground_truth)
            if target is None:
                empty_target_turns[category] += 1
                if user:
                    pending_users.append(user)
                continue
            combined_user = "\n\nFollow-up context:\n".join(pending_users + ([user] if user else []))
            pending_users.clear()
            if combined_user:
                pairs.append((combined_user, target))

        if pairs:
            category_trainable[category] += 1
        else:
            empty_target_rows[category] += 1

        first_pairs = pairs[:1]
        entries.append(
            {
                "id": row["id"],
                "category": category,
                "multi_turn": bool(row.get("multi_turn")),
                "target_pairs": len(pairs),
                "system_text": render(system, []),
                "first_text": render(system, first_pairs),
                "full_text": render(system, pairs),
            }
        )

    for field in ("system_text", "first_text", "full_text"):
        texts = [entry[field] for entry in entries]
        lengths = []
        for start in range(0, len(texts), 128):
            encoded = tokenizer(
                texts[start : start + 128],
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_length=True,
            )
            lengths.extend(encoded["length"])
        for entry, length in zip(entries, lengths):
            entry[field.removesuffix("_text") + "_length"] = int(length)

    by_category: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["category"]].append(entry)
    for category, items in sorted(grouped.items()):
        by_category[category] = {
            "rows": len(items),
            "trainable_rows": category_trainable[category],
            "empty_target_rows": empty_target_rows[category],
            "empty_target_turns": empty_target_turns[category],
            "system_length": summarize([item["system_length"] for item in items]),
            "first_target_length": summarize([item["first_length"] for item in items]),
            "full_length": summarize([item["full_length"] for item in items]),
        }

    multi_items = [entry for entry in entries if entry["multi_turn"]]
    single_items = [entry for entry in entries if not entry["multi_turn"]]
    trainable_items = [entry for entry in entries if entry["target_pairs"] > 0]
    trainable_single_items = [entry for entry in single_items if entry["target_pairs"] > 0]
    trainable_multi_items = [entry for entry in multi_items if entry["target_pairs"] > 0]
    stats = {
        "parquet": str(args.parquet),
        "rows": len(entries),
        "single_turn_rows": len(single_items),
        "multi_turn_rows": len(multi_items),
        "trainable_rows": sum(entry["target_pairs"] > 0 for entry in entries),
        "empty_target_rows": sum(entry["target_pairs"] == 0 for entry in entries),
        "empty_target_turns": sum(empty_target_turns.values()),
        "trainable": {
            "rows": len(trainable_items),
            "system_length": summarize([entry["system_length"] for entry in trainable_items]),
            "first_target_length": summarize([entry["first_length"] for entry in trainable_items]),
            "full_length": summarize([entry["full_length"] for entry in trainable_items]),
        },
        "trainable_single_turn": {
            "rows": len(trainable_single_items),
            "full_length": summarize([entry["full_length"] for entry in trainable_single_items]),
        },
        "trainable_multi_turn": {
            "rows": len(trainable_multi_items),
            "system_length": summarize([entry["system_length"] for entry in trainable_multi_items]),
            "first_target_length": summarize([entry["first_length"] for entry in trainable_multi_items]),
            "full_length": summarize([entry["full_length"] for entry in trainable_multi_items]),
        },
        "all_rows": {
            "system_length": summarize([entry["system_length"] for entry in entries]),
            "first_target_length": summarize([entry["first_length"] for entry in entries]),
            "full_length": summarize([entry["full_length"] for entry in entries]),
        },
        "single_turn": {
            "system_length": summarize([entry["system_length"] for entry in single_items]),
            "full_length": summarize([entry["full_length"] for entry in single_items]),
        },
        "multi_turn": {
            "system_length": summarize([entry["system_length"] for entry in multi_items]),
            "first_target_length": summarize([entry["first_length"] for entry in multi_items]),
            "full_length": summarize([entry["full_length"] for entry in multi_items]),
        },
        "by_category": by_category,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
