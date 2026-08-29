# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


@dataclass
class DataArguments:
    r"""Arguments pertaining to what data we are going to input our model for training and evaluation."""

    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for evaluation. Use commas to separate multiple datasets."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    media_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the folder containing the images, videos or audios. Defaults to `dataset_dir`."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    md_bucket_cutoffs: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated cutoff buckets for MD training, e.g. 1024,2048."},
    )
    md_bucket_k_masks: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated effective k per MD cutoff bucket, e.g. 16,4."},
    )
    md_bucket_train_modes: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated train mode per MD cutoff bucket, e.g. dynamic,dynamic,base."},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    mask_history: bool = field(
        default=False,
        metadata={"help": "Whether or not to mask the history and train on the last turn only."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    buffer_size: int = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    mix_strategy: Literal["concat", "interleave_under", "interleave_over"] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."},
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether or not to ignore the tokens corresponding to the pad label in loss computation."},
    )
    val_size: float = field(
        default=0.0,
        metadata={"help": "Size of the validation set, should be an integer or a float in range `[0,1)`."},
    )
    packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable sequences packing in training. Will automatically enable in pre-training."},
    )
    neat_packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing without cross-attention."},
    )
    tool_format: Optional[str] = field(
        default=None,
        metadata={"help": "Tool format to use for constructing function calling examples."},
    )
    tokenized_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to save or load the tokenized datasets. "
                "If tokenized_path not exists, it will save the tokenized datasets. "
                "If tokenized_path exists, it will load the tokenized datasets."
            )
        },
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.dataset = split_arg(self.dataset)
        self.eval_dataset = split_arg(self.eval_dataset)

        if self.media_dir is None:
            self.media_dir = self.dataset_dir

        if self.md_bucket_cutoffs is not None:
            self.md_bucket_cutoffs = list(map(int, split_arg(self.md_bucket_cutoffs)))

        if self.md_bucket_k_masks is not None:
            self.md_bucket_k_masks = list(map(int, split_arg(self.md_bucket_k_masks)))

        if self.md_bucket_train_modes is not None:
            self.md_bucket_train_modes = [item.lower() for item in split_arg(self.md_bucket_train_modes)]

        if (self.md_bucket_cutoffs is None) != (self.md_bucket_k_masks is None):
            raise ValueError("`md_bucket_cutoffs` and `md_bucket_k_masks` should be specified together.")

        if self.md_bucket_train_modes is not None and self.md_bucket_cutoffs is None:
            raise ValueError("`md_bucket_train_modes` should be specified together with `md_bucket_cutoffs`.")

        if self.md_bucket_cutoffs is not None and self.md_bucket_k_masks is not None:
            if len(self.md_bucket_cutoffs) != len(self.md_bucket_k_masks):
                raise ValueError("`md_bucket_cutoffs` and `md_bucket_k_masks` should have the same length.")

            if any(cutoff <= 0 for cutoff in self.md_bucket_cutoffs):
                raise ValueError("`md_bucket_cutoffs` should contain positive integers.")

            if any(k < 0 for k in self.md_bucket_k_masks):
                raise ValueError("`md_bucket_k_masks` should contain non-negative integers.")

            if any(
                self.md_bucket_cutoffs[i] >= self.md_bucket_cutoffs[i + 1]
                for i in range(len(self.md_bucket_cutoffs) - 1)
            ):
                raise ValueError("`md_bucket_cutoffs` should be strictly increasing.")

            if self.md_bucket_train_modes is None:
                self.md_bucket_train_modes = ["base" if k == 0 else "dynamic" for k in self.md_bucket_k_masks]

            if len(self.md_bucket_train_modes) != len(self.md_bucket_cutoffs):
                raise ValueError("`md_bucket_train_modes` and `md_bucket_cutoffs` should have the same length.")

            allowed_train_modes = {"dynamic", "base"}
            if any(mode not in allowed_train_modes for mode in self.md_bucket_train_modes):
                raise ValueError("`md_bucket_train_modes` should only contain `dynamic` or `base`.")

            for effective_k, train_mode in zip(self.md_bucket_k_masks, self.md_bucket_train_modes):
                if train_mode == "dynamic" and effective_k <= 0:
                    raise ValueError("Dynamic MD buckets should use a positive `k`.")
                if train_mode == "base" and effective_k != 0:
                    raise ValueError("Base MD buckets should use `k=0`.")

        if self.dataset is None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `dataset` is None.")

        if self.eval_dataset is not None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

        if self.interleave_probs is not None:
            if self.mix_strategy == "concat":
                raise ValueError("`interleave_probs` is only valid for interleaved mixing.")

            self.interleave_probs = list(map(float, split_arg(self.interleave_probs)))
            if self.dataset is not None and len(self.dataset) != len(self.interleave_probs):
                raise ValueError("The length of dataset and interleave probs should be identical.")

            if self.eval_dataset is not None and len(self.eval_dataset) != len(self.interleave_probs):
                raise ValueError("The length of eval dataset and interleave probs should be identical.")

        if self.streaming and self.val_size > 1e-6 and self.val_size < 1:
            raise ValueError("Streaming mode should have an integer val size.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")

        if self.mask_history and self.train_on_prompt:
            raise ValueError("`mask_history` is incompatible with `train_on_prompt`.")

        if self.neat_packing:
            self.packing = True

        if self.packing and self.md_bucket_cutoffs is not None:
            raise ValueError("MD bucketed cutoff is not compatible with sequence packing.")

        if self.packing:
            self.cutoff_len -= 1  # avoid pad_to_multiple_of, needs improve

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
