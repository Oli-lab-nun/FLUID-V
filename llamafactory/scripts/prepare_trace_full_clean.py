#!/usr/bin/env python3
"""Clean AgentTraj-L while preserving one complete trajectory per sample."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from transformers import AutoTokenizer


ACTION_RE = re.compile(r"(?im)^\s*Action\s*:")
THOUGHT_RE = re.compile(r"(?im)^\s*Thought\s*:")
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def comparison_text(value: str) -> str:
    return SPACE_RE.sub(" ", value).strip()


def render_nothink(system: str, conversations: list[dict[str, str]]) -> str:
    chunks = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    for index in range(0, len(conversations), 2):
        user = conversations[index]["value"]
        assistant = conversations[index + 1]["value"]
        chunks.append(
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            f"{assistant}<|im_end|>\n"
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
        "over_1600": sum(value > 1600 for value in values),
        "over_2048": sum(value > 2048 for value in values),
        "over_4096": sum(value > 4096 for value in values),
    }


def repetition_metrics(conversations: list[dict[str, str]]) -> tuple[int, int]:
    users = [comparison_text(message["value"]) for message in conversations[0::2]]
    assistants = [comparison_text(message["value"]) for message in conversations[1::2]]
    max_assistant_run = 1
    max_pair_run = 1
    assistant_run = 1
    pair_run = 1
    for index in range(1, len(assistants)):
        if assistants[index] == assistants[index - 1]:
            assistant_run += 1
        else:
            assistant_run = 1
        if (users[index], assistants[index]) == (users[index - 1], assistants[index - 1]):
            pair_run += 1
        else:
            pair_run = 1
        max_assistant_run = max(max_assistant_run, assistant_run)
        max_pair_run = max(max_pair_run, pair_run)
    return max_assistant_run, max_pair_run


def normalize_record(record: dict[str, Any], source: str, record_index: int) -> tuple[dict[str, Any] | None, str | None]:
    messages = record.get("conversations")
    if not isinstance(messages, list) or len(messages) < 4 or len(messages) % 2:
        return None, "invalid_message_count"
    for index, message in enumerate(messages):
        expected_role = "human" if index % 2 == 0 else "gpt"
        if not isinstance(message, dict) or message.get("from") != expected_role:
            return None, "invalid_role_order"
    if messages[1].get("loss") is not False:
        return None, "invalid_bootstrap"
    if any(message.get("loss") is not True for message in messages[3::2]):
        return None, "invalid_target_loss"

    system = clean_text(messages[0].get("value"))
    if not system:
        return None, "empty_system"
    conversations = []
    for index, message in enumerate(messages[2:]):
        value = clean_text(message.get("value"))
        if not value:
            return None, "empty_turn"
        role = "human" if index % 2 == 0 else "gpt"
        if role == "gpt" and not ACTION_RE.search(value):
            return None, "missing_action"
        conversations.append({"from": role, "value": value})

    max_assistant_run, max_pair_run = repetition_metrics(conversations)
    if max_pair_run >= 2:
        return None, "consecutive_duplicate_pair"
    if max_assistant_run >= 3:
        return None, "consecutive_repeated_response"

    assistant_values = [message["value"] for message in conversations[1::2]]
    return {
        "id": f"trace:{source}:{record_index}:{record.get('item_id', 'unknown')}",
        "source": f"trace/{source}",
        "system": system,
        "conversations": conversations,
        "metadata": {
            "trajectory_id": record.get("item_id"),
            "source_record_index": record_index,
            "source_subset": source,
            "assistant_turns": len(assistant_values),
            "thought_turns": sum(bool(THOUGHT_RE.search(value)) for value in assistant_values),
            "action_turns": len(assistant_values),
            "bootstrap_ack_removed": True,
            "full_trajectory": True,
        },
    }, None


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
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_by_source: Counter[str] = Counter()
    kept_by_source: Counter[str] = Counter()
    removed_by_reason: Counter[str] = Counter()
    removed_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    lengths: list[int] = []

    for source_path in sorted(args.trace_dir.glob("*_train.json")):
        source = source_path.stem.removesuffix("_train")
        records = json.loads(source_path.read_text(encoding="utf-8"))
        for record_index, record in enumerate(records):
            raw_by_source[source] += 1
            sample, reason = normalize_record(record, source, record_index)
            if sample is not None:
                key = sample_key(sample)
                if key in seen:
                    reason = "duplicate_trajectory"
                    sample = None
                else:
                    seen.add(key)
            if sample is None:
                reason = reason or "unknown"
                removed_by_reason[reason] += 1
                removed_by_source[source][reason] += 1
                removed.append(
                    {
                        "source": source,
                        "source_record_index": record_index,
                        "trajectory_id": record.get("item_id"),
                        "reason": reason,
                    }
                )
                continue

            token_length = len(
                tokenizer.encode(
                    render_nothink(sample["system"], sample["conversations"]),
                    add_special_tokens=False,
                )
            )
            sample["metadata"]["raw_token_length"] = token_length
            lengths.append(token_length)
            kept_by_source[source] += 1
            kept.append(sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for sample in kept:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp_path.replace(args.output)

    removed_path = args.output.with_suffix(".removed.jsonl")
    removed_temp = removed_path.with_suffix(removed_path.suffix + ".tmp")
    with removed_temp.open("w", encoding="utf-8") as handle:
        for item in removed:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    removed_temp.replace(removed_path)

    stats = {
        "output": str(args.output),
        "removed_output": str(removed_path),
        "raw_trajectories": sum(raw_by_source.values()),
        "written_trajectories": len(kept),
        "removed_trajectories": len(removed),
        "raw_by_source": dict(sorted(raw_by_source.items())),
        "written_by_source": dict(sorted(kept_by_source.items())),
        "removed_by_reason": dict(sorted(removed_by_reason.items())),
        "removed_by_source": {
            source: dict(sorted(reasons.items())) for source, reasons in sorted(removed_by_source.items())
        },
        "assistant_turns": sum(sample["metadata"]["assistant_turns"] for sample in kept),
        "thought_turns": sum(sample["metadata"]["thought_turns"] for sample in kept),
        "action_turns": sum(sample["metadata"]["action_turns"] for sample in kept),
        "token_length": summarize_lengths(lengths),
    }
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
