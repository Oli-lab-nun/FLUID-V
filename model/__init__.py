import os

from .model_nothink_v6_head import MDVLModel as OpenPanguVLNoThinkModelV6Head
from .model_vl_nothink_v6 import MDVLModel as OpenPanguVLNoThinkModelV6


def _maybe_load_openpangu_vl_adapter(model, adapter_path, prefix="OpenPangu-VL"):
    if not adapter_path:
        return model
    if not os.path.isdir(adapter_path):
        print(f"[{prefix}] adapter path not found, skip loading: {adapter_path}")
        return model

    from peft import PeftModel

    print(f"[{prefix}] loading adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    merged_model = peft_model.merge_and_unload()
    print(f"[{prefix}] adapter weights merged into wrapper model.")
    return merged_model


def get_md_model(
    base_model=None,
    k_masks=None,
    is_trainable=False,
    md_model_variant="auto",
    mask_token="<mask>",
    mask_insert_prob=0.25,
    restore_ratio_max=0.5,
    noise_prob=0.05,
    md_train_stage="stage1",
    openpangu_vl_variant="nothink_v6",
    max_train_blocks=None,
    lora_adapter_path=None,
):
    if not (hasattr(base_model, "model") and hasattr(base_model.model, "visual")):
        raise ValueError("FLUID-V OpenPangu-VL wrapper requires a multimodal OpenPangu-VL base model.")

    model_type = getattr(getattr(base_model, "config", None), "model_type", "")
    if model_type != "openpangu_vl":
        raise ValueError(f"Unsupported FLUID-V base model_type: {model_type!r}. Expected 'openpangu_vl'.")

    variant = str(openpangu_vl_variant).lower()
    if variant in {"nothink_v6", "v6", "nothink_v6_full", "v6_full"}:
        wrapper_cls = OpenPanguVLNoThinkModelV6
    elif variant in {"nothink_v6_head", "v6_head", "head"}:
        wrapper_cls = OpenPanguVLNoThinkModelV6Head
    else:
        raise ValueError(
            "Unsupported OpenPangu-VL FLUID-V variant: "
            f"{openpangu_vl_variant!r}. Expected nothink_v6 or nothink_v6_head."
        )

    model = wrapper_cls(base_model=base_model, k_masks=k_masks)
    model = _maybe_load_openpangu_vl_adapter(model, lora_adapter_path, prefix="FLUID-V-OpenPangu-VL")

    if variant in {"nothink_v6_head", "v6_head", "head"}:
        setattr(model, "teacher_adapter_path", lora_adapter_path)
        if hasattr(model, "_enable_k_head_training_only"):
            model._enable_k_head_training_only()
        if hasattr(model, "_assert_k_head_trainable") and is_trainable:
            model._assert_k_head_trainable()

    return model


__all__ = ["get_md_model"]
