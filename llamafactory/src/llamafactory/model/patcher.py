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

from types import MethodType
from typing import TYPE_CHECKING, Any, Optional

import torch
from peft import PeftModel
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase, is_torch_npu_available
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras import logging
from ..extras.misc import infer_optim_dtype, is_env_enabled
from ..extras.packages import is_transformers_version_greater_than
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.kv_cache import configure_kv_cache
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.packing import configure_packing
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import autocast_projector_dtype, configure_visual_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments


logger = logging.get_logger(__name__)


def _patch_glm4v_mrope_config(config: "PretrainedConfig", model: Optional["PreTrainedModel"] = None) -> None:
    r"""Patch GLM4V checkpoints that store mRoPE args as `rope_parameters`.

    Some GLM-4.6V-Flash exports keep `mrope_section` under
    `text_config.rope_parameters`, while the upstream Transformers GLM4V
    attention reads `config.rope_scaling["mrope_section"]` during forward.
    Copying the same values before and after model construction keeps the
    original AR model path intact and avoids touching the shared model files.
    """
    model_type = str(getattr(config, "model_type", "")).lower()
    text_config = getattr(config, "text_config", None)
    if text_config is None and model_type == "glm4v_text":
        text_config = config

    text_model_type = str(getattr(text_config, "model_type", "")).lower() if text_config is not None else ""
    if model_type != "glm4v" and text_model_type != "glm4v_text":
        return

    rope_parameters = getattr(text_config, "rope_parameters", None) if text_config is not None else None
    rope_scaling = getattr(text_config, "rope_scaling", None) if text_config is not None else None
    if rope_scaling is None and rope_parameters is not None:
        rope_scaling = rope_parameters
    if rope_scaling is None:
        rope_scaling = {"mrope_section": [8, 12, 12], "rope_type": "default"}

    try:
        rope_scaling = dict(rope_scaling)
    except TypeError:
        logger.warning_rank0("Cannot patch GLM4V mRoPE config because rope parameters are invalid.")
        return

    rope_scaling.setdefault("mrope_section", [8, 12, 12])
    rope_scaling.setdefault("rope_type", rope_scaling.get("type", "default"))

    if text_config is not None:
        for key in ("partial_rotary_factor", "rope_theta"):
            if key in rope_scaling:
                setattr(text_config, key, rope_scaling[key])

        hidden_size = getattr(text_config, "hidden_size", None)
        num_heads = getattr(text_config, "num_attention_heads", None)
        partial_rotary_factor = getattr(text_config, "partial_rotary_factor", 1.0) or 1.0
        if hidden_size is not None and num_heads:
            head_dim = int(hidden_size) // int(num_heads)
            rotary_dim = int(head_dim * float(partial_rotary_factor))
            mrope_section = list(rope_scaling.get("mrope_section") or [8, 12, 12])
            if sum(mrope_section) * 2 != rotary_dim and sum(mrope_section) * 4 == rotary_dim:
                rope_scaling["mrope_section"] = [int(value) * 2 for value in mrope_section]

    if text_config is not None:
        text_config.rope_scaling = rope_scaling

    language_model = None
    if model is not None:
        language_model = getattr(model, "language_model", None)
        if language_model is None:
            language_model = getattr(getattr(model, "model", None), "language_model", None)

    if language_model is not None:
        if getattr(language_model, "config", None) is not None:
            language_model.config.rope_scaling = rope_scaling
        for layer in getattr(language_model, "layers", []):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None:
                self_attn.rope_scaling = rope_scaling

    logger.info_rank0(f"Patched GLM4V mRoPE config: {rope_scaling}.")


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and tokenizer.model_max_length < model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length  # enlarge the tokenizer max length

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info_rank0("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")


def patch_processor(
    processor: "ProcessorMixin",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", model_args.image_max_pixels)
    setattr(processor, "image_min_pixels", model_args.image_min_pixels)
    setattr(processor, "image_do_pan_and_scan", model_args.image_do_pan_and_scan)
    setattr(processor, "video_max_pixels", model_args.video_max_pixels)
    setattr(processor, "video_min_pixels", model_args.video_min_pixels)
    setattr(processor, "video_fps", model_args.video_fps)
    setattr(processor, "video_maxlen", model_args.video_maxlen)
    setattr(processor, "audio_sampling_rate", model_args.audio_sampling_rate)
    setattr(processor, "use_audio_in_video", model_args.use_audio_in_video)


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    if is_torch_npu_available():
        torch.npu.set_compile_mode(jit_compile=is_env_enabled("JIT_COMPILE"))

    configure_attn_implementation(config, model_args, is_trainable)
    configure_rope(config, model_args, is_trainable)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    _patch_glm4v_mrope_config(config)
    configure_packing(model_args, is_trainable)
    configure_kv_cache(config, model_args, is_trainable)

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "minicpmo":
        setattr(config, "init_audio", True)
        setattr(config, "init_tts", False)

    if "LlavaLlamaForCausalLM" in getattr(config, "architectures", []):
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    if getattr(config, "model_type", None) == "internlm3" and not is_transformers_version_greater_than("4.47.1"):
        raise RuntimeError("InternLM3 model requires transformers>=4.47.1, please upgrade it.")

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # do not cast data type of the model deepspeed zero3 without qlora
    if not (is_deepspeed_zero3_enabled() and model_args.quantization_bit is None):
        init_kwargs["torch_dtype"] = model_args.compute_dtype

        if init_kwargs["low_cpu_mem_usage"] and not is_fsdp_enabled():  # fsdp does not need device map
            if "device_map" not in init_kwargs and model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map  # device map requires low_cpu_mem_usage=True

            if init_kwargs.get("device_map", None) == "auto":
                init_kwargs["offload_folder"] = model_args.offload_folder


def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    _patch_glm4v_mrope_config(model.config, model)

    gen_config = getattr(model, "generation_config", None)  # check and fix generation config
    if gen_config is None:
        try:
            gen_config = GenerationConfig.from_pretrained(model_args.model_name_or_path)
        except Exception:
            try:
                gen_config = GenerationConfig.from_model_config(model.config)
            except Exception:
                gen_config = GenerationConfig()

        model.generation_config = gen_config

    do_sample = getattr(gen_config, "do_sample", False)
    temperature = getattr(gen_config, "temperature", None)
    top_p = getattr(gen_config, "top_p", None)
    typical_p = getattr(gen_config, "typical_p", None)
    if not do_sample and (
        (temperature is not None and temperature != 1.0)
        or (top_p is not None and top_p != 1.0)
        or (typical_p is not None and typical_p != 1.0)
    ):
        setattr(gen_config, "do_sample", True)

    model_type = str(getattr(model.config, "model_type", "")).lower()
    model_generate = getattr(model, "generate", None)
    if (
        model_generate is not None
        and model_type not in ["minicpmv", "minicpmo", "llada", "dream"]
        and "GenerationMixin" not in str(model_generate.__func__)
    ):
        model.generate = MethodType(PreTrainedModel.generate, model)

    if add_valuehead:
        prepare_valuehead_model(model)

    if model_args.resize_vocab:
        resize_embedding_layer(model, tokenizer)

    if is_trainable:
        prepare_model_for_training(model, model_args)
        autocast_projector_dtype(model, model_args)
        add_z3_leaf_module(model)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))
