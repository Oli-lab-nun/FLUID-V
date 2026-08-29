import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _bootstrap_cudnn_library_path() -> None:
    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    library_paths = [
        Path(sys.prefix) / "lib" / version_dir / "site-packages" / "torch" / "lib",
        Path(sys.prefix) / "lib" / version_dir / "site-packages" / "nvidia" / "cudnn" / "lib",
        Path(sys.executable).resolve().parents[1] / "lib" / version_dir / "site-packages" / "torch" / "lib",
        Path(sys.executable).resolve().parents[1] / "lib" / version_dir / "site-packages" / "nvidia" / "cudnn" / "lib",
    ]
    marker = "libcudnn_engines_precompiled.so.9"
    current_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]

    try:
        import ctypes

        ctypes.CDLL(marker)
        return
    except OSError:
        pass

    for library_path in library_paths:
        if not (library_path / marker).is_file():
            continue
        if str(library_path) in current_paths or os.environ.get("_FLUIDV_CUDNN_REEXEC") == "1":
            return

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join([str(library_path), *current_paths])
        env["_FLUIDV_CUDNN_REEXEC"] = "1"
        print(f"[FLUID-V] Re-launching with cuDNN library path: {library_path}", file=sys.stderr)
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


_bootstrap_cudnn_library_path()

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from model.model_nothink_v6_head import MDVLModel as MDVLHeadModel
from model.model_vl_nothink_v6 import MDVLModel


def _sanitize_generation_text(tokenizer, text: str) -> str:
    special_tokens = sorted(set(getattr(tokenizer, "all_special_tokens", [])), key=len, reverse=True)
    for token in special_tokens:
        if token:
            text = text.replace(token, "")
    text = re.sub(r"\[unused\d+\]", "", text)
    return text.strip()


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class FluidVInferenceModel:
    def __init__(
        self,
        base_model_path: str,
        adapter_path: Optional[str],
        head_checkpoint_path: Optional[str],
        device: str,
        torch_dtype: str,
        k_masks: int,
        image_max_pixels: int,
        image_min_pixels: int,
        attn_implementation: Optional[str],
    ) -> None:
        dtype = getattr(torch, torch_dtype)
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        if getattr(config, "model_type", None) == "openpangu_vl":
            model_kwargs["key_mapping"] = {
                "^visual": "model.visual",
                r"^model(?!\.(language_model|visual))": "model.language_model",
            }

        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, config=config, **model_kwargs)
        processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        processor.image_max_pixels = image_max_pixels
        processor.image_min_pixels = image_min_pixels

        wrapper_cls = MDVLHeadModel if head_checkpoint_path else MDVLModel
        md_model = wrapper_cls(base_model=base_model, k_masks=k_masks)
        md_model.set_tokenizer(processor.tokenizer)

        if adapter_path:
            from peft import PeftModel

            md_model = PeftModel.from_pretrained(md_model, adapter_path).merge_and_unload()
            print(f"[FLUID-V] merged adapter: {adapter_path}")

        if head_checkpoint_path:
            md_model = MDVLHeadModel.load_model_for_inference(md_model, head_checkpoint_path)
            print(f"[FLUID-V] loaded dynamic-K head: {head_checkpoint_path}")

        self.device = device
        self.processor = processor
        self.model = md_model.to(device).eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        max_new_tokens: int,
        block_size: int,
        confidence_threshold: float,
        use_standard_generate: bool,
        use_fill_kv: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        content = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt_text], images=list(images), return_tensors="pt")
        inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in inputs.items()}

        started_at = time.perf_counter()
        if use_standard_generate:
            generated_ids = self.model.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            stats: Dict[str, Any] = {}
        else:
            if use_fill_kv:
                if not hasattr(self.model, "generate_fill_kv"):
                    raise NotImplementedError("The loaded FLUID-V model does not implement generate_fill_kv.")
                generate_fn = self.model.generate_fill_kv
            else:
                generate_fn = self.model.generate_dynamic_kv

            result = generate_fn(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                pixel_values_videos=inputs.get("pixel_values_videos"),
                image_grid_thw=inputs.get("image_grid_thw"),
                video_grid_thw=inputs.get("video_grid_thw"),
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                confidence_threshold=confidence_threshold,
                return_stats=True,
            )
            if isinstance(result, (tuple, list)) and len(result) == 2:
                generated_ids, stats = result
            else:
                generated_ids, stats = result, {}

        elapsed = time.perf_counter() - started_at
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][input_len:]
        output_text = self.processor.batch_decode([output_ids], skip_special_tokens=False)[0]
        stats = dict(stats)
        stats.update(
            {
                "wall_time_sec": elapsed,
                "generated_tokens": int(output_ids.shape[0]),
                "num_images": len(images),
            }
        )
        return _sanitize_generation_text(self.processor.tokenizer, output_text), stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLUID-V OpenPangu-VL inference on one prompt and optional images.")
    parser.add_argument("--base-model-path", required=True, help="Path or hub id for the OpenPangu-VL base checkpoint.")
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA adapter checkpoint for FLUID-V stage-1.")
    parser.add_argument("--head-checkpoint-path", default=None, help="Optional stage-2 dynamic-K head checkpoint.")
    parser.add_argument("--image", action="append", default=[], help="Image path. Can be supplied more than once.")
    parser.add_argument("--prompt", required=True, help="User prompt.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default=None, help="Optional transformers attention backend, e.g. sdpa.")
    parser.add_argument("--k-masks", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--image-max-pixels", type=int, default=784000)
    parser.add_argument("--image-min-pixels", type=int, default=3136)
    parser.add_argument("--use-standard-generate", action="store_true", help="Use the base autoregressive generate path.")
    parser.add_argument("--use-fill-kv", action="store_true", help="Use generate_fill_kv instead of generate_dynamic_kv.")
    parser.add_argument("--json", action="store_true", help="Print response and stats as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = [_load_image(path) for path in args.image]
    model = FluidVInferenceModel(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        head_checkpoint_path=args.head_checkpoint_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        k_masks=args.k_masks,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
        attn_implementation=args.attn_implementation,
    )
    response, stats = model.generate(
        prompt=args.prompt,
        images=images,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        confidence_threshold=args.confidence_threshold,
        use_standard_generate=args.use_standard_generate,
        use_fill_kv=args.use_fill_kv,
    )
    if args.json:
        print(json.dumps({"response": response, "stats": stats}, ensure_ascii=False, indent=2))
    else:
        print(response)
        print(json.dumps(stats, ensure_ascii=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
