# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass
import json
import os
import threading
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)

TRAIN_MODE_TO_ID = {"dynamic": 0, "base": 1}


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    _skip_log_lock = threading.Lock()

    def _has_multimodal_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> bool:
        return len(images) > 0 or len(videos) > 0 or len(audios) > 0

    def _get_skip_log_path(self) -> Optional[str]:
        output_dir = os.getenv("MDVL_OUTPUT_DIR", "").strip()
        if not output_dir:
            return None
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, "mdvl_skip_log.jsonl")

    def _append_skip_log(
        self,
        *,
        reason: str,
        cutoff_len: int,
        source_len_before: int,
        source_len_after: int,
        target_len_before: int,
        target_len_after: int,
    ) -> None:
        log_path = self._get_skip_log_path()
        if log_path is None:
            return

        payload = {
            "reason": reason,
            "cutoff_len": int(cutoff_len),
            "source_len_before": int(source_len_before),
            "source_len_after": int(source_len_after),
            "target_len_before": int(target_len_before),
            "target_len_after": int(target_len_after),
        }
        with self._skip_log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _select_md_bucket(self, total_length: int) -> tuple[int, Optional[int], str]:
        bucket_cutoffs = getattr(self.data_args, "md_bucket_cutoffs", None)
        bucket_k_masks = getattr(self.data_args, "md_bucket_k_masks", None)
        bucket_train_modes = getattr(self.data_args, "md_bucket_train_modes", None)
        if not bucket_cutoffs or not bucket_k_masks:
            return int(self.data_args.cutoff_len), None, "dynamic"

        for cutoff, effective_k, train_mode in zip(bucket_cutoffs, bucket_k_masks, bucket_train_modes):
            if total_length <= cutoff:
                return int(cutoff), int(effective_k), str(train_mode)

        return int(bucket_cutoffs[-1]), int(bucket_k_masks[-1]), str(bucket_train_modes[-1])

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> Optional[dict[str, Any]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        has_mm = self._has_multimodal_inputs(images, videos, audios)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        prefix_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        full_length = prefix_length + sum(len(source_ids) + len(target_ids) for source_ids, target_ids in encoded_pairs)
        bucket_cutoff_len, effective_k, train_mode = self._select_md_bucket(full_length)
        total_length = prefix_length

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= bucket_cutoff_len:
                break

            remaining_budget = bucket_cutoff_len - total_length
            source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), remaining_budget)

            # For multimodal samples, truncating the prompt/source can cut image/video
            # placeholder tokens while pixel_values still remain in the batch, which
            # later triggers token/feature mismatch inside the VL backbone.
            # To keep training stable, skip the whole sample whenever a multimodal
            # source segment would be truncated by cutoff_len.
            if has_mm and source_len < len(source_ids):
                logger.warning_rank0(
                    "Dropped multimodal example because source_ids would be truncated "
                    f"({len(source_ids)} -> {source_len}) under cutoff_len={bucket_cutoff_len}."
                )
                self._append_skip_log(
                    reason="multimodal_source_truncated",
                    cutoff_len=bucket_cutoff_len,
                    source_len_before=len(source_ids),
                    source_len_after=source_len,
                    target_len_before=len(target_ids),
                    target_len_after=target_len,
                )
                return None

            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "bucket_cutoff_len": bucket_cutoff_len,
            "effective_k": effective_k,
            "train_mode_id": TRAIN_MODE_TO_ID.get(train_mode, TRAIN_MODE_TO_ID["dynamic"]),
        }

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            encoded = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            if encoded is None:
                continue

            input_ids = encoded["input_ids"]
            labels = encoded["labels"]
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["bucket_cutoff_len"].append(encoded["bucket_cutoff_len"])
            model_inputs["effective_k"].append(encoded["effective_k"])
            model_inputs["train_mode_id"].append(encoded["train_mode_id"])
            model_inputs["images"].append(examples["_images"][i] or [])
            model_inputs["videos"].append(examples["_videos"][i] or [])
            model_inputs["audios"].append(examples["_audios"][i] or [])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            encoded = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            if encoded is None:
                continue

            input_ids = encoded["input_ids"]
            labels = encoded["labels"]
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or [])
            model_inputs["videos"].append(packed_videos or [])
            model_inputs["audios"].append(packed_audios or [])

        return model_inputs
