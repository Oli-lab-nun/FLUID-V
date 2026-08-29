#!/usr/bin/env python3
"""Prepare no-think Trace and BFCLv3 decision-point SFT datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ACTION_RE = re.compile(r"(?im)^\s*Action\s*:\s*")
NO_THINK_TRACE_SUFFIX = (
    "\n\nDo not output analysis or a Thought section. "
    "Respond only with `Action: <your next action>` for each turn."
)
NO_THINK_BFCL_SUFFIX = "\n\nDo not output analysis or a Thought section."
TRUNCATION_MARKER = "\n[... earlier context truncated ...]\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--bfcl-parquet", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff-len", type=int, default=1600)
    parser.add_argument("--max-history-pairs", type=int, default=8)
    return parser.parse_args()


class SampleFitter:
    def __init__(self, tokenizer: Any, cutoff_len: int, max_history_pairs: int) -> None:
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.max_history_pairs = max_history_pairs

    def _token_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def render(self, system: str, conversations: list[dict[str, str]]) -> str:
        chunks = []
        if system:
            chunks.append(f"<|im_start|>system\n{system}<|im_end|>\n")

        for index in range(0, len(conversations), 2):
            user = conversations[index]["value"]
            assistant = conversations[index + 1]["value"]
            chunks.append(
                f"<|im_start|>user\n{user}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                f"{assistant}<|im_end|>\n"
            )

        return "".join(chunks)

    def _truncate_head_tail(self, text: str, max_tokens: int) -> str:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        if max_tokens < 16:
            return self.tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=False)

        marker_ids = self.tokenizer.encode(TRUNCATION_MARKER, add_special_tokens=False)
        content_budget = max(2, max_tokens - len(marker_ids))
        head_len = max(1, (content_budget * 2) // 3)
        tail_len = max(1, content_budget - head_len)
        return (
            self.tokenizer.decode(token_ids[:head_len], skip_special_tokens=False)
            + TRUNCATION_MARKER
            + self.tokenizer.decode(token_ids[-tail_len:], skip_special_tokens=False)
        )

    def fit(
        self, system: str, pairs: list[tuple[str, str]]
    ) -> tuple[str, list[dict[str, str]], dict[str, int]] | None:
        selected = pairs[-self.max_history_pairs :]
        removed_pairs = len(pairs) - len(selected)

        def to_conversations(items: list[tuple[str, str]]) -> list[dict[str, str]]:
            messages = []
            for user, assistant in items:
                messages.append({"from": "human", "value": user})
                messages.append({"from": "gpt", "value": assistant})
            return messages

        conversations = to_conversations(selected)
        token_len = self._token_len(self.render(system, conversations))
        while token_len > self.cutoff_len and len(selected) > 1:
            selected = selected[1:]
            removed_pairs += 1
            conversations = to_conversations(selected)
            token_len = self._token_len(self.render(system, conversations))

        system_truncated = 0
        user_truncated = 0
        if token_len > self.cutoff_len:
            current_user, current_target = selected[-1]
            empty_len = self._token_len(self.render("", to_conversations([(current_user, current_target)])))
            system_len = self._token_len(system)
            excess = token_len - self.cutoff_len
            new_system_budget = max(128, system_len - excess)
            if new_system_budget < system_len:
                system = self._truncate_head_tail(system, new_system_budget)
                system_truncated = 1
                token_len = self._token_len(self.render(system, conversations))

            if token_len > self.cutoff_len:
                user_len = self._token_len(current_user)
                excess = token_len - self.cutoff_len
                new_user_budget = max(64, user_len - excess)
                if new_user_budget < user_len:
                    current_user = self._truncate_head_tail(current_user, new_user_budget)
                    selected[-1] = (current_user, current_target)
                    conversations = to_conversations(selected)
                    user_truncated = 1
                    token_len = self._token_len(self.render(system, conversations))

            if token_len > self.cutoff_len or empty_len > self.cutoff_len:
                return None

        return system, conversations, {
            "token_length": token_len,
            "history_pairs": len(selected) - 1,
            "removed_history_pairs": removed_pairs,
            "system_truncated": system_truncated,
            "user_truncated": user_truncated,
        }


def extract_action(text: str) -> str | None:
    matches = list(ACTION_RE.finditer(text))
    if not matches:
        return None
    action = text[matches[-1].end() :].strip()
    return f"Action: {action}" if action else None


def sample_key(sample: dict[str, Any]) -> str:
    payload = json.dumps(
        {"system": sample["system"], "conversations": sample["conversations"]},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl_atomic(path: Path, samples: Iterable[dict[str, Any]]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    seen = set()
    written = 0
    duplicates = 0
    with temp_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            key = sample_key(sample)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    temp_path.replace(path)
    return written, duplicates


def prepare_trace(
    trace_dir: Path, fitter: SampleFitter, output_path: Path
) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    per_source: Counter[str] = Counter()
    token_lengths: list[int] = []

    def generate() -> Iterable[dict[str, Any]]:
        for source_path in sorted(trace_dir.glob("*_train.json")):
            source = source_path.stem.removesuffix("_train")
            records = json.loads(source_path.read_text(encoding="utf-8"))
            stats["trajectories"] += len(records)
            for record in records:
                raw_messages = record.get("conversations", [])
                if len(raw_messages) < 4 or raw_messages[0].get("from") != "human":
                    stats["invalid_trajectories"] += 1
                    continue

                system = raw_messages[0].get("value", "").strip() + NO_THINK_TRACE_SUFFIX
                pairs: list[tuple[str, str]] = []
                for index in range(2, len(raw_messages) - 1, 2):
                    user = raw_messages[index]
                    assistant = raw_messages[index + 1]
                    if user.get("from") != "human" or assistant.get("from") != "gpt":
                        stats["invalid_turns"] += 1
                        continue
                    if assistant.get("loss") is not True:
                        stats["non_supervised_turns"] += 1
                        continue

                    action = extract_action(str(assistant.get("value", "")))
                    if action is None:
                        stats["missing_action_targets"] += 1
                        continue

                    pairs.append((str(user.get("value", "")).strip(), action))
                    fitted = fitter.fit(system, list(pairs))
                    if fitted is None:
                        stats["overlength_dropped"] += 1
                        continue
                    fitted_system, conversations, fit_stats = fitted
                    stats["system_truncated"] += fit_stats["system_truncated"]
                    stats["user_truncated"] += fit_stats["user_truncated"]
                    stats["history_pairs_removed"] += fit_stats["removed_history_pairs"]
                    token_lengths.append(fit_stats["token_length"])
                    per_source[source] += 1
                    yield {
                        "id": f"trace:{source}:{record.get('item_id', 'unknown')}:{len(pairs) - 1}",
                        "source": f"trace/{source}",
                        "system": fitted_system,
                        "conversations": conversations,
                        "metadata": {
                            "trajectory_id": record.get("item_id"),
                            "decision_index": len(pairs) - 1,
                            **fit_stats,
                        },
                    }

    written, duplicates = write_jsonl_atomic(output_path, generate())
    return {
        "output": str(output_path),
        "written": written,
        "duplicates_removed": duplicates,
        "per_source": dict(sorted(per_source.items())),
        "token_length": summarize_lengths(token_lengths),
        **dict(stats),
    }


def format_function_calls(ground_truth: Any) -> str | None:
    if not isinstance(ground_truth, list) or not ground_truth:
        return None
    calls = [str(call).strip() for call in ground_truth if str(call).strip()]
    return "[" + ", ".join(calls) + "]" if calls else None


def prepare_bfcl(parquet_path: Path, fitter: SampleFitter, output_path: Path) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    per_category: Counter[str] = Counter()
    token_lengths: list[int] = []

    def generate() -> Iterable[dict[str, Any]]:
        table = pq.read_table(parquet_path)
        stats["repository_rows"] = table.num_rows
        for row in table.to_pylist():
            if not row.get("multi_turn"):
                stats["non_v3_rows_skipped"] += 1
                continue

            stats["v3_rows"] += 1
            category = str(row.get("test_category", "unknown"))
            groups = json.loads(row["turns"])
            ground_truths = json.loads(row["ground_truth"])
            if not isinstance(groups, list) or not isinstance(ground_truths, list) or len(groups) != len(ground_truths):
                stats["invalid_rows"] += 1
                continue

            system = ""
            pairs: list[tuple[str, str]] = []
            pending_missing_param: list[str] = []
            for turn_index, (group, ground_truth) in enumerate(zip(groups, ground_truths)):
                group = group if isinstance(group, list) else []
                for message in group:
                    if message.get("role") == "system" and not system:
                        system = str(message.get("content", "")).strip() + NO_THINK_BFCL_SUFFIX
                users = [str(message.get("content", "")).strip() for message in group if message.get("role") == "user"]
                user_text = "\n\n".join(text for text in users if text)
                target = format_function_calls(ground_truth)
                if target is None:
                    stats["empty_ground_truth_turns_skipped"] += 1
                    if category == "multi_turn_miss_param" and user_text:
                        pending_missing_param.append(user_text)
                    continue

                if pending_missing_param:
                    user_text = "\n\nFollow-up context:\n".join(pending_missing_param + ([user_text] if user_text else []))
                    pending_missing_param.clear()
                    stats["missing_parameter_turns_merged"] += 1
                if not user_text:
                    stats["missing_user_turns"] += 1
                    continue

                pairs.append((user_text, target))
                fitted = fitter.fit(system, list(pairs))
                if fitted is None:
                    stats["overlength_dropped"] += 1
                    continue
                fitted_system, conversations, fit_stats = fitted
                stats["system_truncated"] += fit_stats["system_truncated"]
                stats["user_truncated"] += fit_stats["user_truncated"]
                stats["history_pairs_removed"] += fit_stats["removed_history_pairs"]
                token_lengths.append(fit_stats["token_length"])
                per_category[category] += 1
                yield {
                    "id": f"bfcl_v3:{row['id']}:{turn_index}",
                    "source": "bfcl_v3_multiturn_evaluation",
                    "system": fitted_system,
                    "conversations": conversations,
                    "metadata": {
                        "bfcl_id": row["id"],
                        "turn_index": turn_index,
                        "category": category,
                        "benchmark_evaluation_data": True,
                        **fit_stats,
                    },
                }

    written, duplicates = write_jsonl_atomic(output_path, generate())
    return {
        "output": str(output_path),
        "written": written,
        "duplicates_removed": duplicates,
        "per_category": dict(sorted(per_category.items())),
        "token_length": summarize_lengths(token_lengths),
        **dict(stats),
    }


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {}
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def main() -> None:
    args = parse_args()
    if args.cutoff_len <= 0 or args.max_history_pairs <= 0:
        raise ValueError("cutoff and history limits must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    fitter = SampleFitter(tokenizer, args.cutoff_len, args.max_history_pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_output = args.output_dir / f"trace_agenttraj_nothink_{args.cutoff_len}.jsonl"
    bfcl_output = args.output_dir / f"bfcl_v3_multiturn_nothink_{args.cutoff_len}.jsonl"

    stats = {
        "cutoff_len": args.cutoff_len,
        "max_history_pairs": args.max_history_pairs,
        "trace": prepare_trace(args.trace_dir, fitter, trace_output),
        "bfcl_v3": prepare_bfcl(args.bfcl_parquet, fitter, bfcl_output),
    }
    stats_path = args.output_dir / "preparation_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
