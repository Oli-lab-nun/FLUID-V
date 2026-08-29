# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
import time
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        logger.info_rank0("Starting optimizer creation in CustomSeq2SeqTrainer.")
        optimizer = super().create_optimizer()
        if optimizer is not None and hasattr(optimizer, "param_groups"):
            before = len(optimizer.param_groups)
            optimizer.param_groups = [group for group in optimizer.param_groups if len(group.get("params", [])) > 0]
            after = len(optimizer.param_groups)
            if before != after:
                logger.warning_rank0(
                    "Removed empty optimizer param_groups to keep scheduler in sync (%d -> %d).",
                    before,
                    after,
                )
        logger.info_rank0("Finished optimizer creation in CustomSeq2SeqTrainer.")
        return optimizer

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        logger.info_rank0("Starting scheduler creation in CustomSeq2SeqTrainer.")
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        scheduler = super().create_scheduler(num_training_steps, optimizer)
        if self.optimizer is not None and scheduler is not None and hasattr(self.optimizer, "param_groups"):
            base_lrs = getattr(scheduler, "base_lrs", None)
            if base_lrs is not None and len(base_lrs) != len(self.optimizer.param_groups):
                logger.warning_rank0(
                    "Adjusting lr scheduler base_lrs to match optimizer param_groups (%d -> %d).",
                    len(base_lrs),
                    len(self.optimizer.param_groups),
                )
                scheduler.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
                if hasattr(scheduler, "_last_lr"):
                    scheduler._last_lr = list(scheduler.base_lrs)
        logger.info_rank0("Finished scheduler creation in CustomSeq2SeqTrainer.")
        return scheduler

    @override
    def get_train_dataloader(self):
        logger.info_rank0("Building train dataloader in CustomSeq2SeqTrainer.")
        dataloader = super().get_train_dataloader()
        logger.info_rank0("Finished building train dataloader in CustomSeq2SeqTrainer.")
        return dataloader

    @override
    def _wrap_model(self, model, training=True, dataloader=None):
        logger.info_rank0("Entering _wrap_model in CustomSeq2SeqTrainer.")
        wrapped_model = super()._wrap_model(model, training=training, dataloader=dataloader)
        logger.info_rank0("Finished _wrap_model in CustomSeq2SeqTrainer.")
        return wrapped_model

    @override
    def _get_train_sampler(self, dataset=None) -> Optional["torch.utils.data.Sampler"]:
        train_dataset = dataset if dataset is not None else self.train_dataset
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(train_dataset)

        try:
            return super()._get_train_sampler(dataset)
        except TypeError:
            return super()._get_train_sampler()

    def _is_diffusion_mask_model(self, model: "torch.nn.Module") -> bool:
        unwrapped = model.module if hasattr(model, "module") else model
        candidates = [
            unwrapped,
            getattr(unwrapped, "base_model", None),
            getattr(getattr(unwrapped, "base_model", None), "model", None),
        ]
        for candidate in candidates:
            config = getattr(candidate, "config", None)
            if str(getattr(config, "model_type", "")).lower() in {"llada", "dream"}:
                return True

        return False

    def _compute_diffusion_mask_loss(self, model: "torch.nn.Module", inputs: dict[str, Any]) -> "torch.Tensor":
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        supervised = labels.ne(IGNORE_INDEX)
        if not bool(supervised.any().item()):
            return super().compute_loss(model, inputs)

        config = getattr(model, "config", None)
        model_type = str(getattr(config, "model_type", "")).lower()
        mask_token_id = getattr(config, "mask_token_id", None)
        if mask_token_id is None:
            unwrapped = model.module if hasattr(model, "module") else model
            unwrapped_config = getattr(unwrapped, "config", None)
            model_type = str(getattr(unwrapped_config, "model_type", model_type)).lower()
            mask_token_id = getattr(unwrapped_config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("LLaDA training requires `config.mask_token_id`.")

        mask_prob = float(os.environ.get("DIFFUSION_SFT_MASK_PROB", os.environ.get("LLADA_SFT_MASK_PROB", "0.5")))
        mask_prob = min(max(mask_prob, 1e-6), 1.0)
        sampled = torch.rand(input_ids.shape, device=input_ids.device).lt(mask_prob)
        mask_positions = sampled & supervised

        rows_without_mask = supervised.any(dim=1) & ~mask_positions.any(dim=1)
        if bool(rows_without_mask.any().item()):
            for row_idx in torch.nonzero(rows_without_mask, as_tuple=False).flatten().tolist():
                candidate_positions = torch.nonzero(supervised[row_idx], as_tuple=False).flatten()
                chosen = candidate_positions[
                    torch.randint(candidate_positions.numel(), (1,), device=input_ids.device)
                ]
                mask_positions[row_idx, chosen] = True

        masked_input_ids = input_ids.masked_fill(mask_positions, int(mask_token_id))
        attention_mask = inputs.get("attention_mask")
        attention_bias = None
        if attention_mask is not None and attention_mask.dim() > 2:
            attention_bias = attention_mask
            attention_mask = None
        elif attention_mask is not None and model_type == "dream":
            attention_mask = attention_mask.to(torch.bool)[:, None, None, :]

        forward_kwargs = {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
        }
        if attention_bias is not None:
            forward_kwargs["attention_bias"] = attention_bias

        outputs = model(**forward_kwargs)
        loss_labels = labels.masked_fill(~mask_positions, IGNORE_INDEX)
        return F.cross_entropy(
            outputs.logits.float().view(-1, outputs.logits.size(-1)),
            loss_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        rank = os.environ.get("RANK", "0")
        should_log = rank in {"0", "-1"} and not getattr(self, "_llamafactory_logged_first_compute_loss", False)
        start = time.perf_counter() if should_log else None
        if should_log:
            logger.info_rank0("Entering first compute_loss.")

        if self._is_diffusion_mask_model(model):
            loss = self._compute_diffusion_mask_loss(model, inputs)
        else:
            loss = super().compute_loss(model, inputs, *args, **kwargs)

        if should_log:
            logger.info_rank0("Finished first compute_loss in %.2fs.", time.perf_counter() - start)
            self._llamafactory_logged_first_compute_loss = True

        return loss

    @override
    def training_step(self, model, inputs, num_items_in_batch=None):
        rank = os.environ.get("RANK", "0")
        should_log = rank in {"0", "-1"} and not getattr(self, "_llamafactory_logged_first_training_step", False)
        start = time.perf_counter() if should_log else None
        if should_log:
            logger.info_rank0("Entering first training_step.")

        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        if should_log:
            logger.info_rank0("Finished first training_step in %.2fs.", time.perf_counter() - start)
            self._llamafactory_logged_first_training_step = True

        return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
