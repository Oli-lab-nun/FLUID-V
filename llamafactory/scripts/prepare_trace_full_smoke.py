#!/usr/bin/env python3
"""Select full, overlength AgentTraj-L traces for stage-two smoke tests."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-tokens", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=100)
    return parser.parse_args()


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


def normalize_trace(record: dict[str, Any], source: str, record_index: int) -> dict[str, Any] | None:
    messages = record.get("conversations", [])
    if len(messages) < 4 or messages[0].get("from") != "human":
        return None
    if messages[1].get("from") != "gpt" or messages[1].get("loss") is not False:
        return None

    conversations = []
    for index, message in enumerate(messages[2:]):
        expected_role = "human" if index % 2 == 0 else "gpt"
        if message.get("from") != expected_role:
            return None
        if expected_role == "gpt" and message.get("loss") is not True:
            return None
        conversations.append({"from": expected_role, "value": str(message.get("value", ""))})

    if not conversations or conversations[-1]["from"] != "gpt":
        return None
    return {
        "id": f"trace:{source}:{record_index}:{record.get('item_id', 'unknown')}",
        "source": f"trace/{source}",
        "system": str(messages[0].get("value", "")),
        "conversations": conversations,
        "metadata": {
            "trajectory_id": record.get("item_id"),
            "source_record_index": record_index,
            "source_subset": source,
            "thoughts_preserved": True,
            "bootstrap_ack_removed": True,
        },
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid = 0

    for source_path in sorted(args.trace_dir.glob("*_train.json")):
        source = source_path.stem.removesuffix("_train")
        records = json.loads(source_path.read_text(encoding="utf-8"))
        for record_index, record in enumerate(records):
            sample = normalize_trace(record, source, record_index)
            if sample is None:
                invalid += 1
                continue
            token_length = len(
                tokenizer.encode(
                    render_nothink(sample["system"], sample["conversations"]),
                    add_special_tokens=False,
                )
            )
            if token_length <= args.min_tokens:
                continue
            sample["metadata"]["raw_token_length"] = token_length
            candidates[source].append(sample)

    queues = {}
    for source, samples in candidates.items():
        samples.sort(key=lambda item: (-item["metadata"]["raw_token_length"], item["id"]))
        queues[source] = deque(samples)

    selected = []
    source_names = sorted(queues)
    while len(selected) < args.num_samples and any(queues[source] for source in source_names):
        for source in source_names:
            if queues[source] and len(selected) < args.num_samples:
                selected.append(queues[source].popleft())

    if len(selected) != args.num_samples:
        raise ValueError(f"requested {args.num_samples} samples, found {len(selected)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for sample in selected:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp_path.replace(args.output)

    selected_by_source = defaultdict(int)
    lengths = []
    for sample in selected:
        selected_by_source[sample["metadata"]["source_subset"]] += 1
        lengths.append(sample["metadata"]["raw_token_length"])
    stats = {
        "output": str(args.output),
        "selected": len(selected),
        "selection_threshold": f"> {args.min_tokens}",
        "invalid_records": invalid,
        "available_over_threshold": {key: len(value) for key, value in sorted(candidates.items())},
        "selected_by_source": dict(sorted(selected_by_source.items())),
        "selected_length_min": min(lengths),
        "selected_length_max": max(lengths),
    }
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
