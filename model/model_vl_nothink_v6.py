import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional, Tuple

from transformers.cache_utils import DynamicCache
from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class MDVLModelV6(PreTrainedModel, GenerationMixin):
    """
    v6: decoded-token conditioned middle-state refinement with embedding MSE.

    Training uses no-noise random restore to expose the model to decoded-token
    context inside a block. A single rollout produces confidence estimates:
    confident slots switch to token embeddings, while unresolved slots keep
    middle states. Inference stays simple and does not add a separate mask graph
    correction pass.

    Compared with v5, v6 removes the residual branch from middle-state updates
    and adds a small embedding-space MSE loss on the refined middle states.
    """

    config_class = PretrainedConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def __init__(
        self,
        base_model: PreTrainedModel,
        k_masks: Optional[int] = None,
        restore_ratio: float = 0.3,
        train_confidence_threshold: float = 0.9,
        loss_decay_gamma: float = 7.0,
        **kwargs,
    ):
        config = base_model.config
        super().__init__(config)

        self.model = base_model
        if hasattr(self.model, "get_output_embeddings"):
            self.lm_head = self.model.get_output_embeddings()
        else:
            self.lm_head = getattr(self.model, "lm_head", None)
        if self.lm_head is None:
            raise ValueError("基础模型必须暴露 lm_head 或 output embeddings。")

        self.tokenizer = None
        self.mask_token_id = -1
        self.generation_config = getattr(self.model, "generation_config", None)

        self.hidden_size = self._infer_hidden_size()
        self.vocab_size = self._infer_vocab_size()
        self.max_k = int(k_masks) if k_masks is not None else 8
        if self.max_k <= 0:
            raise ValueError(f"k_masks 必须大于 0，当前值为 {self.max_k}。")

        self.restore_ratio = float(restore_ratio)
        self.train_confidence_threshold = float(train_confidence_threshold)
        self.loss_decay_gamma = float(loss_decay_gamma)

        self.base_every = 1
        self.recurrent_every = 1
        self.schedule_window = 2
        self._train_forward_counter = 0

        self.block_ratio = 0.3
        self.loss_chunk_size = 2048

        self.soft_token_topk = max(1, min(4, self.max_k))
        self.soft_token_temperature = 1.0
        self.entropy_gate_floor = 0.5
        self.middle_gate_min = 0.05
        self.middle_gate_max = 0.95

        self.semantic_topk = self.soft_token_topk
        self.semantic_temperature = self.soft_token_temperature

        self.middle_state_norm = nn.LayerNorm(self.hidden_size)
        self.ce_loss_weight = 0.95
        self.embedding_mse_weight = 0.05

        logger.info(
            "[OpenPangu-VL][MDVL-V6] decoded-token conditioned training enabled; "
            "restore=%.2f, train_tau=%.2f, loss_gamma=%.2f, ce=%.2f, emb_mse=%.2f.",
            self.restore_ratio,
            self.train_confidence_threshold,
            self.loss_decay_gamma,
            self.ce_loss_weight,
            self.embedding_mse_weight,
        )

    def _infer_hidden_size(self) -> int:
        config_values = [
            getattr(self.config, "hidden_size", None),
            getattr(getattr(self.config, "text_config", None), "hidden_size", None),
            getattr(getattr(self.model, "config", None), "hidden_size", None),
            getattr(getattr(getattr(self.model, "config", None), "text_config", None), "hidden_size", None),
        ]
        for value in config_values:
            if value is not None:
                return int(value)

        input_embeddings = self.model.get_input_embeddings()
        if input_embeddings is None:
            raise ValueError("无法从配置或词嵌入中推断 hidden_size。")
        return int(input_embeddings.weight.shape[1])

    def _infer_vocab_size(self) -> int:
        if hasattr(self.lm_head, "weight"):
            return int(self.lm_head.weight.shape[0])
        if getattr(self.config, "vocab_size", None) is not None:
            return int(self.config.vocab_size)
        text_cfg = getattr(self.config, "text_config", None)
        if text_cfg is not None and getattr(text_cfg, "vocab_size", None) is not None:
            return int(text_cfg.vocab_size)
        raise ValueError("无法推断词表大小。")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            for token in ("<|fim_pad|>", "[unused33]", "[MASK]"):
                if token in tokenizer.get_vocab():
                    mask_token_id = tokenizer.convert_tokens_to_ids(token)
                    break

        if mask_token_id is None or mask_token_id == tokenizer.unk_token_id:
            raise ValueError("当前 tokenizer 没有可用的 mask token。")

        image_token_id = getattr(self.config, "image_token_id", None)
        video_token_id = getattr(self.config, "video_token_id", None)
        if mask_token_id in {image_token_id, video_token_id}:
            raise ValueError(
                f"mask token {mask_token_id} 与视觉占位 token 冲突：image={image_token_id}, video={video_token_id}。"
            )

        self.mask_token_id = int(mask_token_id)
        logger.info("[OpenPangu-VL][MDVL-V6] mask_token_id 已设置为 %s", self.mask_token_id)

    def _get_mm_forbidden_token_ids(self) -> List[int]:
        token_ids: List[int] = []
        for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
            token_id = getattr(self.config, name, None)
            if token_id is not None:
                token_ids.append(int(token_id))
        uniq: List[int] = []
        seen = set()
        for token_id in token_ids:
            if token_id not in seen and token_id >= 0:
                uniq.append(token_id)
                seen.add(token_id)
        return uniq

    def _mask_mm_forbidden_logits(self, logits: torch.Tensor) -> torch.Tensor:
        forbidden_ids = self._get_mm_forbidden_token_ids()
        if not forbidden_ids:
            return logits

        vocab_size = int(logits.size(-1))
        valid_ids = [token_id for token_id in forbidden_ids if 0 <= token_id < vocab_size]
        if not valid_ids:
            return logits

        masked_logits = logits.clone()
        min_value = torch.finfo(masked_logits.dtype).min
        masked_logits.index_fill_(
            -1,
            torch.tensor(valid_ids, device=masked_logits.device, dtype=torch.long),
            min_value,
        )
        return masked_logits

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits_2d = logits.reshape(-1, logits.size(-1))
        labels_1d = labels.reshape(-1)
        valid_count = (labels_1d != -100).sum()
        if int(valid_count.item()) == 0:
            return logits_2d.sum() * 0.0

        if self.loss_chunk_size > 0 and logits_2d.size(0) > self.loss_chunk_size:
            total_loss = torch.zeros((), device=logits_2d.device, dtype=torch.float32)
            total_count = torch.zeros((), device=labels_1d.device, dtype=torch.long)
            for start in range(0, logits_2d.size(0), self.loss_chunk_size):
                end = start + self.loss_chunk_size
                chunk_logits = logits_2d[start:end].float()
                chunk_labels = labels_1d[start:end]
                total_loss = total_loss + F.cross_entropy(
                    chunk_logits,
                    chunk_labels,
                    ignore_index=-100,
                    reduction="sum",
                )
                total_count = total_count + (chunk_labels != -100).sum()
            return total_loss / total_count.clamp(min=1).to(total_loss.dtype)

        return F.cross_entropy(logits_2d.float(), labels_1d, ignore_index=-100)

    def _compute_weighted_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits_2d = logits.reshape(-1, logits.size(-1))
        labels_1d = labels.reshape(-1)
        valid = labels_1d != -100
        if not bool(valid.any().item()):
            return logits_2d.sum() * 0.0

        per_token = F.cross_entropy(
            logits_2d.float(),
            labels_1d,
            ignore_index=-100,
            reduction="none",
        )
        if weights is None:
            weight_1d = valid.to(per_token.dtype)
        else:
            weight_1d = weights.reshape(-1).to(device=per_token.device, dtype=per_token.dtype)
            weight_1d = torch.where(valid, weight_1d, torch.zeros_like(weight_1d))

        return (per_token * weight_1d).sum() / weight_1d.sum().clamp_min(1.0)

    def _ensure_loss_has_grad(self, loss: torch.Tensor) -> torch.Tensor:
        if loss.requires_grad:
            return loss

        for param in self.parameters():
            if param.requires_grad and param.numel() > 0:
                anchor = param.reshape(-1)[0].to(device=loss.device, dtype=loss.dtype)
                return loss + anchor * loss.new_zeros(())

        return loss

    def _restore_random_mask_inputs(
        self,
        *,
        new_input_ids: torch.Tensor,
        source_input_ids: torch.Tensor,
        restore_labels: torch.Tensor,
        col_indices: torch.Tensor,
        target_indices: torch.Tensor,
        valid_target: torch.Tensor,
        is_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        restored_mask = torch.zeros_like(new_input_ids, dtype=torch.bool)
        if not bool(is_mask.any().item()):
            return new_input_ids, restored_mask

        batch_size = new_input_ids.shape[0]
        device = new_input_ids.device
        seq_len = source_input_ids.shape[1]

        mask_positions = is_mask.unsqueeze(0).expand(batch_size, -1)
        restore_ratio = max(0.0, min(1.0, self.restore_ratio))
        element_probs = torch.rand_like(new_input_ids, dtype=torch.float32, device=device)
        should_restore = mask_positions & (element_probs < restore_ratio)
        valid_restore = should_restore & valid_target.unsqueeze(0) & (restore_labels != -100)
        if not bool(valid_restore.any().item()):
            return new_input_ids, restored_mask

        safe_target_indices = target_indices.clamp(max=seq_len - 1)
        real_tokens = source_input_ids[:, safe_target_indices]

        blocked_restore_ids = []
        for name in ("image_token_id", "video_token_id"):
            token_id = getattr(self.config, name, None)
            if token_id is not None and int(token_id) >= 0:
                blocked_restore_ids.append(int(token_id))
        if self.mask_token_id >= 0:
            blocked_restore_ids.append(self.mask_token_id)

        if blocked_restore_ids:
            blocked_tensor = torch.tensor(blocked_restore_ids, device=device, dtype=real_tokens.dtype)
            valid_restore = valid_restore & (~torch.isin(real_tokens, blocked_tensor))
            if not bool(valid_restore.any().item()):
                return new_input_ids, restored_mask

        restored_mask = valid_restore
        return torch.where(valid_restore, real_tokens, new_input_ids), restored_mask

    def _resolve_effective_k(self, effective_k: Optional[torch.Tensor]) -> int:
        if effective_k is None:
            return self.max_k

        if torch.is_tensor(effective_k):
            if effective_k.numel() == 0:
                return self.max_k
            effective_k = effective_k.reshape(-1)[0].item()

        effective_k = int(effective_k)
        return max(1, min(effective_k, self.max_k))

    def _build_dynamick_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        effective_k: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        if batch_size != 1:
            raise NotImplementedError("MDVLModelV6 的 dynamic-k 训练当前只支持 per-device batch_size=1。")
        effective_k = self._resolve_effective_k(effective_k)

        shifted_labels = torch.roll(labels, shifts=-1, dims=1)
        shifted_labels[:, -1] = -100

        is_response_col = (shifted_labels != -100).all(dim=0)
        rand_probs = torch.rand(seq_len, device=device)
        remask_block_mask = is_response_col & (rand_probs < self.block_ratio)

        k_counts = torch.zeros(seq_len, dtype=torch.long, device=device)
        if remask_block_mask.any():
            sampled = torch.randint(
                low=0,
                high=effective_k + 1,
                size=(seq_len,),
                device=device,
                dtype=torch.long,
            )
            k_counts = torch.where(remask_block_mask, sampled, k_counts)

        if is_response_col.any() and not bool((k_counts > 0).any().item()):
            first_valid = torch.nonzero(is_response_col, as_tuple=False)[0, 0]
            k_counts[first_valid] = 1

        repeats = 1 + k_counts
        col_indices = torch.repeat_interleave(torch.arange(seq_len, device=device), repeats)
        new_len = int(col_indices.shape[0])

        cumsum_repeats = torch.cumsum(repeats, dim=0)
        block_starts = torch.cat([torch.tensor([0], device=device), cumsum_repeats[:-1]])
        range_indices = torch.arange(new_len, device=device)
        offsets = range_indices - block_starts[col_indices]
        is_mask = offsets > 0

        new_input_ids = torch.full(
            (batch_size, new_len),
            self.mask_token_id,
            dtype=input_ids.dtype,
            device=device,
        )
        expanded_inputs = input_ids[:, col_indices]
        new_input_ids = torch.where(is_mask.unsqueeze(0), new_input_ids, expanded_inputs)

        target_indices = col_indices + offsets
        in_bounds_target = target_indices < seq_len
        safe_target_indices = target_indices.clamp(max=seq_len - 1)
        expanded_targets = shifted_labels[:, safe_target_indices]
        has_supervised_target = (expanded_targets != -100).all(dim=0)
        valid_target = in_bounds_target & has_supervised_target
        new_labels = expanded_targets.masked_fill(~valid_target.unsqueeze(0), -100)

        if attention_mask is None:
            base_attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        else:
            base_attention_mask = attention_mask.to(device)
        new_attention_mask_2d = base_attention_mask[:, col_indices]

        block_ids_per_col = torch.full((seq_len,), -1, dtype=torch.long, device=device)
        block_start_cols = torch.nonzero(k_counts > 0, as_tuple=False).squeeze(-1)
        num_blocks = int(block_start_cols.numel())
        if num_blocks > 0:
            block_ids_per_col[block_start_cols] = torch.arange(num_blocks, device=device)

        block_ids = block_ids_per_col[col_indices]
        block_slots = offsets - 1
        block_slots = torch.where(is_mask, block_slots, torch.full_like(block_slots, -1))

        active_slot_positions = (
            is_mask
            & (block_ids >= 0)
            & (block_slots >= 0)
            & (block_slots < self.max_k)
            & valid_target
        )
        active_slot_mask = torch.zeros((num_blocks, self.max_k), dtype=torch.bool, device=device)
        if num_blocks > 0 and bool(active_slot_positions.any().item()):
            active_block_ids = block_ids[active_slot_positions]
            active_block_slots = block_slots[active_slot_positions]
            active_slot_mask[active_block_ids, active_block_slots] = True

        q_block = block_ids.unsqueeze(1)
        k_block = block_ids.unsqueeze(0)
        q_is_mask = is_mask.unsqueeze(1)
        k_is_mask = is_mask.unsqueeze(0)
        q_step = block_slots.unsqueeze(1)
        k_step = block_slots.unsqueeze(0)
        q_col = col_indices.unsqueeze(1)
        k_col = col_indices.unsqueeze(0)

        attn_bias = torch.full((new_len, new_len), float("-inf"), device=device)
        main_key_visible = (~k_is_mask) & (k_col <= q_col)
        mask_key_visible = q_is_mask & k_is_mask & (q_block == k_block) & (k_step <= q_step)
        valid_connections = main_key_visible | mask_key_visible
        attn_bias.masked_fill_(valid_connections, 0.0)
        new_attention_mask_4d = attn_bias.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

        if attention_mask is not None:
            pad_mask = new_attention_mask_2d == 0
            new_attention_mask_4d = new_attention_mask_4d.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(-1),
                float("-inf"),
            )
            new_attention_mask_4d = new_attention_mask_4d.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            row_all_inf = torch.isneginf(new_attention_mask_4d).all(dim=-1).squeeze(1)
            if row_all_inf.any():
                batch_idx, pos_idx = torch.where(row_all_inf)
                new_attention_mask_4d[batch_idx, 0, pos_idx, pos_idx] = 0.0

        new_input_ids, restored_mask = self._restore_random_mask_inputs(
            new_input_ids=new_input_ids,
            source_input_ids=input_ids,
            restore_labels=new_labels,
            col_indices=col_indices,
            target_indices=target_indices,
            valid_target=valid_target,
            is_mask=is_mask,
        )

        middle_slot_mask = active_slot_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()
        if num_blocks > 0 and bool(active_slot_positions.any().item()):
            active_block_ids = block_ids[active_slot_positions]
            active_block_slots = block_slots[active_slot_positions]
            restored_active = restored_mask[:, active_slot_positions]
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(restored_active)
            expanded_block_ids = active_block_ids.unsqueeze(0).expand_as(restored_active)
            expanded_block_slots = active_block_slots.unsqueeze(0).expand_as(restored_active)
            middle_slot_mask[batch_indices, expanded_block_ids, expanded_block_slots] = ~restored_active

        meta = {
            "block_start_cols": block_start_cols,
            "num_blocks": torch.tensor(num_blocks, device=device),
            "col_indices": col_indices,
            "offsets": offsets,
            "target_indices": target_indices,
            "is_mask": is_mask,
            "block_ids": block_ids,
            "block_slots": block_slots,
            "valid_target": valid_target,
            "k_counts": k_counts,
            "effective_k": torch.tensor(effective_k, device=device),
            "active_slot_mask": active_slot_mask,
            "middle_slot_mask": middle_slot_mask,
            "restored_mask": restored_mask,
        }
        return new_input_ids, new_labels, new_attention_mask_4d, new_attention_mask_2d, meta

    def _update_middle_state(
        self,
        current_middle_state: torch.Tensor,
        block_hidden: torch.Tensor,
        block_logits: torch.Tensor,
        active_slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        compute_device = current_middle_state.device
        compute_dtype = current_middle_state.dtype
        soft_embeddings = self._build_semantic_embeddings(block_logits).to(
            device=compute_device,
            dtype=compute_dtype,
        )
        block_hidden = block_hidden.to(device=compute_device, dtype=compute_dtype)
        topk = min(self.soft_token_topk, block_logits.size(-1))
        topk_values = torch.topk(block_logits.float(), k=topk, dim=-1).values
        topk_probs = torch.softmax(topk_values / self.soft_token_temperature, dim=-1)

        confidence = topk_probs.amax(dim=-1, keepdim=True)
        if topk > 1:
            entropy = -(topk_probs * torch.log(topk_probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
            entropy_confidence = 1.0 - entropy / torch.log(
                torch.tensor(float(topk), device=block_logits.device, dtype=topk_probs.dtype)
            )
        else:
            entropy_confidence = torch.ones_like(confidence)

        entropy_gate = self.entropy_gate_floor + (1.0 - self.entropy_gate_floor) * entropy_confidence
        confidence = (confidence * entropy_gate).clamp(self.middle_gate_min, self.middle_gate_max)
        confidence = confidence.to(device=current_middle_state.device, dtype=current_middle_state.dtype).detach()

        denoise_blend = (1.0 - confidence) * current_middle_state + confidence * soft_embeddings
        next_state = self.middle_state_norm(denoise_blend)

        active_state_mask = active_slot_mask.to(device=next_state.device, dtype=torch.bool)
        if active_state_mask.ndim == 2:
            active_state_mask = active_state_mask.unsqueeze(0).expand(next_state.size(0), -1, -1)
        active_state_mask = active_state_mask.unsqueeze(-1)
        return torch.where(active_state_mask, next_state, current_middle_state)

    def _get_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask_2d: torch.Tensor,
        meta: Dict[str, torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
    ):
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "get_rope_index"):
            return None

        base_position_ids, rope_deltas = self.model.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask_2d,
        )
        self.model.model.rope_deltas = rope_deltas

        col_indices = meta["col_indices"]
        offsets = meta["offsets"]
        seq_len = input_ids.shape[1]
        target_indices = (col_indices + offsets).clamp(max=seq_len - 1)

        if base_position_ids.ndim == 3:
            expanded_position_ids = base_position_ids[:, :, target_indices]
        else:
            expanded_position_ids = base_position_ids[:, target_indices]
        return expanded_position_ids

    def _model_core_forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ):
        return self.model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
        )

    def _build_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        pixel_values_videos: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0)
            image_token_id = int(self.config.image_token_id)
            num_image_tokens = int((input_ids == image_token_id).sum().item())
            num_image_features = int(image_embeds.shape[0])
            if num_image_tokens != num_image_features:
                raise ValueError(
                    "OpenPangu-VL 图像特征与图像占位 token 数量不一致："
                    f"tokens={num_image_tokens}, features={num_image_features}"
                )

            image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.model.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0)
            video_token_id = int(self.config.video_token_id)
            num_video_tokens = int((input_ids == video_token_id).sum().item())
            num_video_features = int(video_embeds.shape[0])
            if num_video_tokens != num_video_features:
                raise ValueError(
                    "OpenPangu-VL 视频特征与视频占位 token 数量不一致："
                    f"tokens={num_video_tokens}, features={num_video_features}"
                )

            video_mask = (input_ids == video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return inputs_embeds

    def _output_weight(self) -> torch.Tensor:
        if hasattr(self.lm_head, "weight"):
            return self.lm_head.weight
        input_embeddings = self.get_input_embeddings()
        if input_embeddings is None:
            raise ValueError("无法获取输出词嵌入矩阵。")
        return input_embeddings.weight

    def _input_weight(self) -> torch.Tensor:
        input_embeddings = self.get_input_embeddings()
        if input_embeddings is None or not hasattr(input_embeddings, "weight"):
            raise ValueError("无法获取输入词嵌入矩阵。")
        return input_embeddings.weight

    def _mask_embedding(self) -> torch.Tensor:
        if self.mask_token_id < 0:
            raise ValueError("请先调用 model.set_tokenizer(tokenizer)。")
        return self._input_weight()[self.mask_token_id].detach()

    def _initial_block_state(
        self,
        batch_size: int,
        num_blocks: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask_embedding = self._mask_embedding().to(device=device, dtype=dtype)
        return mask_embedding.view(1, 1, 1, -1).expand(batch_size, num_blocks, self.max_k, -1).clone()

    def _build_semantic_embeddings(self, logits: torch.Tensor) -> torch.Tensor:
        topk = min(self.semantic_topk, logits.size(-1))
        topk_values, topk_ids = torch.topk(logits, k=topk, dim=-1)
        topk_probs = torch.softmax(topk_values / self.semantic_temperature, dim=-1)
        token_weight = self._input_weight().detach()
        gathered_embeddings = token_weight[topk_ids]
        return torch.sum(topk_probs.unsqueeze(-1) * gathered_embeddings, dim=-2)

    def _top1_confidence_and_embeddings(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(logits.float(), dim=-1)
        confidence, token_ids = torch.max(probs, dim=-1)
        token_weight = self._input_weight().detach()
        token_embeddings = token_weight[token_ids].to(device=logits.device)
        return confidence, token_embeddings

    def _extract_block_tensors(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        meta: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        mask_positions = meta["is_mask"]
        num_blocks = int(meta["num_blocks"].item())
        if num_blocks == 0 or not bool(mask_positions.any().item()):
            return None, None, None

        batch_size, _, hidden_size = hidden_states.shape
        vocab_size = logits.size(-1)
        masked_hidden = hidden_states[:, mask_positions, :]
        masked_logits = logits[:, mask_positions, :]
        masked_labels = labels[:, mask_positions] if labels is not None else None

        block_ids = meta["block_ids"][mask_positions]
        block_slots = meta["block_slots"][mask_positions]
        valid_target = meta["valid_target"][mask_positions]
        valid_mask = (block_ids >= 0) & (block_slots >= 0) & (block_slots < self.max_k) & valid_target
        if not bool(valid_mask.any().item()):
            return None, None, None

        valid_block_ids = block_ids[valid_mask]
        valid_block_slots = block_slots[valid_mask]
        sparse_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

        base_hidden = self._mask_embedding().to(hidden_states.device, hidden_states.dtype)
        block_hidden = base_hidden.view(1, 1, 1, -1).expand(
            batch_size, num_blocks, self.max_k, hidden_size
        ).clone()

        base_logits = self.lm_head(base_hidden.view(1, 1, -1)).squeeze(0).squeeze(0)
        block_logits = base_logits.view(1, 1, 1, -1).expand(
            batch_size, num_blocks, self.max_k, vocab_size
        ).clone()

        block_hidden[:, valid_block_ids, valid_block_slots, :] = masked_hidden[:, sparse_indices, :]
        block_logits[:, valid_block_ids, valid_block_slots, :] = masked_logits[:, sparse_indices, :]

        if masked_labels is None:
            block_labels = None
        else:
            block_labels = torch.full(
                (batch_size, num_blocks, self.max_k),
                -100,
                dtype=masked_labels.dtype,
                device=masked_labels.device,
            )
            block_labels[:, valid_block_ids, valid_block_slots] = masked_labels[:, sparse_indices]

        return block_hidden, block_logits, block_labels

    def _extract_input_token_block_tensors(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        meta: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        mask_positions = meta["is_mask"]
        num_blocks = int(meta["num_blocks"].item())
        if num_blocks == 0 or not bool(mask_positions.any().item()):
            return None, None

        batch_size, _, hidden_size = hidden_states.shape
        vocab_size = logits.size(-1)
        mask_indices = torch.nonzero(mask_positions, as_tuple=False).squeeze(-1)

        block_ids = meta["block_ids"][mask_positions]
        block_slots = meta["block_slots"][mask_positions]
        valid_target = meta["valid_target"][mask_positions]
        valid_mask = (block_ids >= 0) & (block_slots >= 0) & (block_slots < self.max_k) & valid_target
        if not bool(valid_mask.any().item()):
            return None, None

        valid_block_ids = block_ids[valid_mask]
        valid_block_slots = block_slots[valid_mask]
        active_positions = mask_indices[valid_mask]
        predictor_positions = (active_positions - 1).clamp_min(0)

        base_hidden = self._mask_embedding().to(hidden_states.device, hidden_states.dtype)
        block_hidden = base_hidden.view(1, 1, 1, -1).expand(
            batch_size, num_blocks, self.max_k, hidden_size
        ).clone()

        base_logits = self.lm_head(base_hidden.view(1, 1, -1)).squeeze(0).squeeze(0)
        block_logits = base_logits.view(1, 1, 1, -1).expand(
            batch_size, num_blocks, self.max_k, vocab_size
        ).clone()

        block_hidden[:, valid_block_ids, valid_block_slots, :] = hidden_states[:, predictor_positions, :]
        block_logits[:, valid_block_ids, valid_block_slots, :] = logits[:, predictor_positions, :]
        return block_hidden, block_logits

    def _inject_block_state_into_embeddings(
        self,
        base_embeddings: torch.Tensor,
        block_state: torch.Tensor,
        meta: Dict[str, torch.Tensor],
        decoded_token_embeddings: Optional[torch.Tensor] = None,
        decoded_slot_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        injected_embeddings = base_embeddings.clone()
        mask_positions = meta["is_mask"]
        mask_indices = torch.nonzero(mask_positions, as_tuple=False).squeeze(-1)
        block_ids = meta["block_ids"][mask_positions]
        block_slots = meta["block_slots"][mask_positions]
        valid_target = meta["valid_target"][mask_positions]
        valid_mask = (block_ids >= 0) & (block_slots >= 0) & (block_slots < self.max_k) & valid_target
        if not bool(valid_mask.any().item()):
            return injected_embeddings

        sparse_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        valid_block_ids = block_ids[valid_mask]
        valid_block_slots = block_slots[valid_mask]
        active_positions = mask_indices[sparse_indices]
        active_block_state = block_state[:, valid_block_ids, valid_block_slots, :].to(injected_embeddings.dtype)

        if decoded_token_embeddings is not None and decoded_slot_mask is not None:
            decoded_embeddings = decoded_token_embeddings[
                :, valid_block_ids, valid_block_slots, :
            ].to(injected_embeddings.dtype)
            decoded_active = decoded_slot_mask[:, valid_block_ids, valid_block_slots].to(dtype=torch.bool)
            if bool(decoded_active.any().item()):
                batch_idx, active_idx = torch.where(decoded_active)
                decoded_positions = active_positions[active_idx]
                injected_embeddings[batch_idx, decoded_positions, :] = decoded_embeddings[batch_idx, active_idx, :]

        restored_mask = meta.get("restored_mask", None)
        if restored_mask is None and decoded_slot_mask is None:
            injected_embeddings[:, active_positions, :] = active_block_state
            return injected_embeddings

        if restored_mask is None:
            unresolved_active = torch.ones(
                (injected_embeddings.size(0), active_positions.numel()),
                device=injected_embeddings.device,
                dtype=torch.bool,
            )
        else:
            restored_active = restored_mask[:, active_positions]
            unresolved_active = ~restored_active

        if decoded_slot_mask is not None:
            decoded_active = decoded_slot_mask[:, valid_block_ids, valid_block_slots].to(
                device=injected_embeddings.device,
                dtype=torch.bool,
            )
            unresolved_active = unresolved_active & (~decoded_active)

        if not bool(unresolved_active.any().item()):
            return injected_embeddings

        batch_idx, active_idx = torch.where(unresolved_active)
        unresolved_positions = active_positions[active_idx]
        injected_embeddings[batch_idx, unresolved_positions, :] = active_block_state[batch_idx, active_idx, :]
        return injected_embeddings

    def _build_block_target_embeddings(
        self,
        source_input_ids: torch.Tensor,
        meta: Dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        mask_positions = meta["is_mask"]
        num_blocks = int(meta["num_blocks"].item())
        if num_blocks == 0 or not bool(mask_positions.any().item()):
            return None, None

        mask_indices = torch.nonzero(mask_positions, as_tuple=False).squeeze(-1)
        block_ids = meta["block_ids"][mask_positions]
        block_slots = meta["block_slots"][mask_positions]
        valid_target = meta["valid_target"][mask_positions]
        valid_mask = (block_ids >= 0) & (block_slots >= 0) & (block_slots < self.max_k) & valid_target
        if not bool(valid_mask.any().item()):
            return None, None

        valid_block_ids = block_ids[valid_mask]
        valid_block_slots = block_slots[valid_mask]
        sparse_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        active_positions = mask_indices[sparse_indices]

        target_indices = meta["target_indices"][active_positions].clamp(max=source_input_ids.shape[1] - 1)
        target_token_ids = source_input_ids[:, target_indices]
        target_embeddings = self._input_weight().detach()[target_token_ids].to(device=device, dtype=dtype)

        block_target_embeddings = torch.zeros(
            source_input_ids.size(0),
            num_blocks,
            self.max_k,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )
        block_target_mask = torch.zeros(
            source_input_ids.size(0),
            num_blocks,
            self.max_k,
            device=device,
            dtype=torch.bool,
        )
        block_target_embeddings[:, valid_block_ids, valid_block_slots, :] = target_embeddings
        block_target_mask[:, valid_block_ids, valid_block_slots] = True
        return block_target_embeddings, block_target_mask

    def _compute_embedding_mse_loss(
        self,
        predicted_state: torch.Tensor,
        target_embeddings: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        active = target_mask.to(device=predicted_state.device, dtype=torch.bool)
        if active.ndim == 2:
            active = active.unsqueeze(0).expand(predicted_state.shape[:3])

        per_slot = F.mse_loss(
            predicted_state.float(),
            target_embeddings.to(device=predicted_state.device, dtype=predicted_state.dtype).float(),
            reduction="none",
        ).mean(dim=-1)
        return per_slot[active].mean() if bool(active.any().item()) else per_slot.sum() * 0.0

    def _build_direct_decode_loss_weights(
        self,
        labels: torch.Tensor,
        meta: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights = torch.where(labels != -100, weights, torch.zeros_like(weights))
        if self.loss_decay_gamma <= 0:
            return weights

        mask_positions = meta["is_mask"]
        if not bool(mask_positions.any().item()):
            return weights

        block_slots = meta["block_slots"][mask_positions]
        valid_target = meta["valid_target"][mask_positions]
        valid_mask = (block_slots >= 0) & (block_slots < self.max_k) & valid_target
        if not bool(valid_mask.any().item()):
            return weights

        slot_ids = block_slots[valid_mask].to(dtype=torch.float32, device=labels.device)
        pos_weights = torch.exp(-slot_ids / float(self.loss_decay_gamma))
        mask_indices = torch.nonzero(mask_positions, as_tuple=False).squeeze(-1)[valid_mask]
        weights[:, mask_indices] = weights[:, mask_indices] * pos_weights.unsqueeze(0)
        return weights

    def _merge_refined_logits(
        self,
        base_logits: torch.Tensor,
        refined_block_logits: torch.Tensor,
        meta: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        merged_logits = base_logits.clone()
        mask_positions = meta["is_mask"]
        mask_indices = torch.nonzero(mask_positions, as_tuple=False).squeeze(-1)
        block_ids = meta["block_ids"][mask_positions]
        block_slots = meta["block_slots"][mask_positions]
        valid_target = meta["valid_target"][mask_positions]
        valid_mask = (block_ids >= 0) & (block_slots >= 0) & (block_slots < self.max_k) & valid_target
        if not bool(valid_mask.any().item()):
            return merged_logits

        sparse_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        valid_block_ids = block_ids[valid_mask]
        valid_block_slots = block_slots[valid_mask]
        active_positions = mask_indices[sparse_indices]
        merged_logits[:, active_positions, :] = refined_block_logits[:, valid_block_ids, valid_block_slots, :]
        return merged_logits

    def _select_iteration_mode(self) -> str:
        mode = "base" if (self._train_forward_counter % self.schedule_window) < self.base_every else "recurrent"
        self._train_forward_counter += 1
        return mode

    def _run_base_training_step(
        self,
        *,
        new_input_ids: torch.Tensor,
        new_labels: torch.Tensor,
        attention_mask_4d: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor],
        pixel_values_videos: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        meta: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        base_embeddings = self._build_multimodal_embeddings(
            input_ids=new_input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        outputs = self._model_core_forward(
            inputs_embeds=base_embeddings,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        direct_weights = self._build_direct_decode_loss_weights(new_labels, meta)
        base_loss = self._compute_weighted_loss(logits, new_labels, direct_weights)
        zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
        return logits, base_loss, {
            "spatial_ce": zero,
            "embedding_mse": zero,
            "refine_loss": zero,
        }

    def _run_recurrent_refinement_training(
        self,
        *,
        source_input_ids: torch.Tensor,
        new_input_ids: torch.Tensor,
        new_labels: torch.Tensor,
        attention_mask_4d: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor],
        pixel_values_videos: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        meta: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        def _fallback_to_base():
            output_logits, base_loss, _ = self._run_base_training_step(
                new_input_ids=new_input_ids,
                new_labels=new_labels,
                attention_mask_4d=attention_mask_4d,
                position_ids=position_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                meta=meta,
            )
            zero = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
            return output_logits, base_loss, {
                "spatial_ce": base_loss,
                "embedding_mse": zero,
                "refine_loss": base_loss,
            }

        num_blocks = int(meta["num_blocks"].item())
        if num_blocks == 0:
            return _fallback_to_base()

        base_embeddings = self._build_multimodal_embeddings(
            input_ids=new_input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        has_vision_inputs = pixel_values is not None or pixel_values_videos is not None

        fallback_to_base = False
        input_block_hidden = None
        input_block_logits = None
        init_confidence = None
        init_token_embeddings = None
        unresolved_slot_mask = None
        decoded_slot_mask = None

        with torch.no_grad():
            init_outputs = self._model_core_forward(
                inputs_embeds=base_embeddings,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
            )
            init_hidden = init_outputs.last_hidden_state
            init_logits_all = self.lm_head(init_hidden)
            input_block_hidden, input_block_logits = self._extract_input_token_block_tensors(
                init_hidden,
                init_logits_all,
                meta,
            )
            if input_block_hidden is None:
                fallback_to_base = True
            else:
                if has_vision_inputs:
                    input_block_logits = self._mask_mm_forbidden_logits(input_block_logits)

                init_confidence, init_token_embeddings = self._top1_confidence_and_embeddings(input_block_logits)
                unresolved_slot_mask = meta["middle_slot_mask"] & (
                    init_confidence < float(self.train_confidence_threshold)
                )
                decoded_slot_mask = meta["middle_slot_mask"] & (~unresolved_slot_mask)

                if not bool(unresolved_slot_mask.any().item()):
                    fallback_to_base = True

        if fallback_to_base:
            return _fallback_to_base()

        current_middle_state = self._initial_block_state(
            batch_size=new_input_ids.size(0),
            num_blocks=num_blocks,
            device=input_block_hidden.device,
            dtype=input_block_hidden.dtype,
        )
        rollout_state = self._update_middle_state(
            current_middle_state,
            input_block_hidden,
            input_block_logits,
            unresolved_slot_mask,
        )

        refine_embeddings = self._inject_block_state_into_embeddings(
            base_embeddings,
            rollout_state,
            meta,
            decoded_token_embeddings=init_token_embeddings,
            decoded_slot_mask=decoded_slot_mask,
        )
        final_outputs = self._model_core_forward(
            inputs_embeds=refine_embeddings,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
        )
        final_hidden = final_outputs.last_hidden_state
        final_logits_all = self.lm_head(final_hidden)
        final_block_hidden, final_block_logits, block_labels = self._extract_block_tensors(
            final_hidden,
            final_logits_all,
            new_labels,
            meta,
        )
        if final_block_hidden is None:
            return _fallback_to_base()

        if has_vision_inputs:
            final_block_logits = self._mask_mm_forbidden_logits(final_block_logits)

        main_ce = self._compute_loss(final_block_logits, block_labels)
        target_embeddings, target_embedding_mask = self._build_block_target_embeddings(
            source_input_ids,
            meta,
            device=rollout_state.device,
            dtype=rollout_state.dtype,
        )
        if target_embeddings is None or target_embedding_mask is None:
            embedding_mse = main_ce * 0.0
        else:
            mse_mask = target_embedding_mask & unresolved_slot_mask.to(device=target_embedding_mask.device)
            embedding_mse = self._compute_embedding_mse_loss(
                rollout_state,
                target_embeddings,
                mse_mask,
            )
        total_refine_loss = self.ce_loss_weight * main_ce + self.embedding_mse_weight * embedding_mse
        refined_logits = self._merge_refined_logits(final_logits_all, final_block_logits, meta)
        zero_base = torch.zeros((), device=refined_logits.device, dtype=refined_logits.dtype)
        return refined_logits, zero_base, {
            "spatial_ce": main_ce,
            "embedding_mse": embedding_mse,
            "refine_loss": total_refine_loss,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        effective_k: Optional[torch.Tensor] = None,
        train_mode_id: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.mask_token_id == -1:
            raise ValueError("请先调用 model.set_tokenizer(tokenizer)。")

        if labels is None:
            return self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                **kwargs,
            )

        (
            new_input_ids,
            new_labels,
            new_attention_mask_4d,
            new_attention_mask_2d,
            meta,
        ) = self._build_dynamick_batch(input_ids, labels, attention_mask, effective_k=effective_k)

        position_ids = self._get_position_ids(
            input_ids=input_ids,
            attention_mask_2d=attention_mask if attention_mask is not None else torch.ones_like(input_ids),
            meta=meta,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        iteration_mode = self._select_iteration_mode() if self.training else "recurrent"
        if iteration_mode == "base":
            output_logits, base_loss, aux_losses = self._run_base_training_step(
                new_input_ids=new_input_ids,
                new_labels=new_labels,
                attention_mask_4d=new_attention_mask_4d,
                position_ids=position_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                meta=meta,
            )
            total_loss = base_loss
        else:
            output_logits, base_loss, aux_losses = self._run_recurrent_refinement_training(
                source_input_ids=input_ids,
                new_input_ids=new_input_ids,
                new_labels=new_labels,
                attention_mask_4d=new_attention_mask_4d,
                position_ids=position_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                meta=meta,
            )
            total_loss = aux_losses["refine_loss"]

        total_loss = self._ensure_loss_has_grad(total_loss)
        return CausalLMOutputWithPast(loss=total_loss, logits=output_logits)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self.model.set_output_embeddings(new_embeddings)

    @torch.no_grad()
    def generate_dynamic_kv(
        self,
        input_ids: torch.LongTensor,
        eos_token_id: int,
        max_new_tokens: int = 128,
        block_size: int = 16,
        confidence_threshold: float = 0.9,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        """
        KV-cache block generation for v6.

        Diffusion generate denoises high-confidence prefix tokens immediately. Remaining
        mask states are updated in-place for the next diffusion step.
        """
        if self.mask_token_id == -1:
            raise ValueError("请先调用 model.set_tokenizer(tokenizer)。")
        if input_ids.size(0) != 1:
            raise NotImplementedError("generate_dynamic_kv 当前只支持 batch_size=1，避免批量 EOS/decode 裁剪出错。")

        device = input_ids.device
        initial_len = input_ids.shape[1]
        current_ids = input_ids.clone()
        current_attention_mask = (
            attention_mask.clone() if attention_mask is not None else torch.ones_like(current_ids, device=device)
        )
        fixed_block = max(1, min(int(block_size), self.max_k))
        has_vision_inputs = pixel_values is not None or pixel_values_videos is not None
        finished = False
        prefill_forward_calls = 0
        diffusion_generate_calls = 0
        refine_generate_calls = 0
        cache_decode_forward_calls = 0
        decoded_token_total = 0

        def _extend_attention_mask(base_mask: Optional[torch.Tensor], total_len: int) -> Optional[torch.Tensor]:
            if base_mask is None:
                return None
            if base_mask.shape[1] == total_len:
                return base_mask
            extra = total_len - base_mask.shape[1]
            if extra <= 0:
                return base_mask[:, :total_len]
            pad = torch.ones((base_mask.shape[0], extra), device=base_mask.device, dtype=base_mask.dtype)
            return torch.cat([base_mask, pad], dim=1)

        def _longest_confident_prefix_len(token_conf: torch.Tensor, threshold: float) -> int:
            confident = token_conf >= float(threshold)
            k = confident.size(1)
            if confident.size(0) == 1:
                row = confident[0]
                first_false = (~row).nonzero(as_tuple=False)
                prefix_len = int(first_false[0].item()) if first_false.numel() else k
                return max(1, prefix_len)

            prefix_len = 0
            for j in range(k):
                if bool(confident[:, j].all().item()):
                    prefix_len += 1
                else:
                    break
            return max(1, prefix_len)

        def _build_generation_meta(block_len: int) -> Dict[str, torch.Tensor]:
            active_slot_mask = torch.zeros((1, self.max_k), dtype=torch.bool, device=device)
            active_slot_mask[0, :block_len] = True
            return {"middle_slot_mask": active_slot_mask}

        def _shift_middle_state(state: torch.Tensor, consumed_len: int, remain_len: int) -> torch.Tensor:
            shifted = self._initial_block_state(
                batch_size=state.size(0),
                num_blocks=state.size(1),
                device=state.device,
                dtype=state.dtype,
            )
            if remain_len > 0:
                shifted[:, :, :remain_len, :] = state[:, :, consumed_len : consumed_len + remain_len, :]
            return shifted

        def _build_segment_embeddings(
            segment_ids: torch.Tensor,
            middle_state: Optional[torch.Tensor],
            middle_start: int,
            middle_len: int,
        ) -> torch.Tensor:
            segment_embeddings = self.get_input_embeddings()(segment_ids)
            if middle_state is not None and middle_len > 0:
                segment_embeddings = segment_embeddings.clone()
                segment_embeddings[:, middle_start : middle_start + middle_len, :] = middle_state[
                    :, 0, :middle_len, :
                ].to(segment_embeddings.dtype)
            return segment_embeddings

        def _pack_block_tensors(
            denoise_hidden: torch.Tensor,
            denoise_logits: torch.Tensor,
            block_len: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            hidden_size = int(denoise_hidden.shape[-1])
            vocab_size = int(denoise_logits.shape[-1])
            base_hidden = self._mask_embedding().to(denoise_hidden.device, denoise_hidden.dtype)
            block_hidden = base_hidden.view(1, 1, 1, -1).expand(
                current_ids.size(0), 1, self.max_k, hidden_size
            ).clone()
            block_hidden[:, 0, :block_len, :] = denoise_hidden[:, :block_len, :]

            block_logits = torch.zeros(
                (current_ids.size(0), 1, self.max_k, vocab_size),
                device=denoise_hidden.device,
                dtype=denoise_logits.dtype,
            )
            block_logits[:, 0, :block_len, :] = denoise_logits[:, :block_len, :]
            return block_hidden, block_logits

        def _run_generate_segment(
            segment_ids: torch.Tensor,
            middle_state: Optional[torch.Tensor],
            middle_start: int,
            middle_len: int,
            past_key_values: DynamicCache,
            prefix_len: int,
        ):
            segment_len = int(segment_ids.shape[1])
            full_attention_mask = _extend_attention_mask(current_attention_mask, prefix_len + segment_len)
            cache_position = torch.arange(prefix_len, prefix_len + segment_len, device=device)
            segment_embeddings = _build_segment_embeddings(segment_ids, middle_state, middle_start, middle_len)
            return self.model(
                input_ids=None,
                attention_mask=full_attention_mask,
                inputs_embeds=segment_embeddings,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                output_hidden_states=True,
            )

        def _decode_tokens(
            decoded_ids: torch.Tensor,
            past_key_values: DynamicCache,
            prefix_len: int,
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            decode_attention_mask = _extend_attention_mask(current_attention_mask, prefix_len + decoded_ids.shape[1])
            cache_position = torch.arange(prefix_len, prefix_len + decoded_ids.shape[1], device=device)
            outputs = self.model(
                input_ids=decoded_ids,
                attention_mask=decode_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                output_hidden_states=True,
            )
            return outputs.logits[:, -1:, :], outputs.hidden_states[-1][:, -1:, :], decode_attention_mask

        past_key_values = DynamicCache()
        prefill_cache_position = torch.arange(initial_len, device=device)
        prefill_outputs = self.model(
            input_ids=current_ids,
            attention_mask=current_attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=prefill_cache_position,
            output_hidden_states=True,
        )
        prefill_forward_calls += 1
        current_prefix_logit = prefill_outputs.logits[:, -1:, :]
        current_prefix_hidden = prefill_outputs.hidden_states[-1][:, -1:, :]
        if has_vision_inputs:
            current_prefix_logit = self._mask_mm_forbidden_logits(current_prefix_logit)

        while (current_ids.shape[1] - initial_len) < max_new_tokens:
            remaining_budget = max_new_tokens - (current_ids.shape[1] - initial_len)
            cur_block = min(fixed_block, remaining_budget)
            unresolved_len = cur_block
            current_middle_state = self._initial_block_state(
                batch_size=current_ids.size(0),
                num_blocks=1,
                device=device,
                dtype=self._mask_embedding().dtype,
            )
            decoded_in_block = torch.empty(
                (current_ids.size(0), 0),
                dtype=current_ids.dtype,
                device=device,
            )

            while unresolved_len > 0:
                prefix_len = int(past_key_values.get_seq_length())
                decoded_len = int(decoded_in_block.shape[1])
                mask_ids = torch.full(
                    (current_ids.size(0), unresolved_len),
                    self.mask_token_id,
                    dtype=current_ids.dtype,
                    device=device,
                )
                segment_ids = mask_ids if decoded_len == 0 else torch.cat([decoded_in_block, mask_ids], dim=1)
                meta = _build_generation_meta(unresolved_len)

                diffusion_outputs = _run_generate_segment(
                    segment_ids=segment_ids,
                    middle_state=current_middle_state,
                    middle_start=decoded_len,
                    middle_len=unresolved_len,
                    past_key_values=past_key_values,
                    prefix_len=prefix_len,
                )
                diffusion_generate_calls += 1

                diffusion_hidden = diffusion_outputs.hidden_states[-1]
                diffusion_logits = diffusion_outputs.logits
                bridge_logit = (
                    current_prefix_logit
                    if decoded_len == 0
                    else diffusion_logits[:, decoded_len - 1 : decoded_len, :]
                )
                bridge_hidden = (
                    current_prefix_hidden
                    if decoded_len == 0
                    else diffusion_hidden[:, decoded_len - 1 : decoded_len, :]
                )
                suffix_hidden = diffusion_hidden[:, decoded_len:, :]
                suffix_logits = diffusion_logits[:, decoded_len:, :]
                if has_vision_inputs:
                    bridge_logit = self._mask_mm_forbidden_logits(bridge_logit)
                    suffix_logits = self._mask_mm_forbidden_logits(suffix_logits)

                denoise_logits = torch.cat([bridge_logit, suffix_logits], dim=1)[:, :unresolved_len, :]
                denoise_hidden = torch.cat([bridge_hidden, suffix_hidden], dim=1)[:, :unresolved_len, :]
                probs = torch.softmax(denoise_logits.float(), dim=-1)
                token_conf, token_ids = torch.max(probs, dim=-1)
                decode_len = _longest_confident_prefix_len(token_conf, confidence_threshold)
                decoded_ids = token_ids[:, :decode_len].to(current_ids.dtype)

                past_key_values.crop(prefix_len)

                if eos_token_id is not None:
                    is_eos = decoded_ids == eos_token_id
                    if is_eos.any():
                        first_eos_idx = int(is_eos.nonzero(as_tuple=True)[1].min().item())
                        decoded_until_eos = decoded_ids[:, : first_eos_idx + 1]
                        decoded_in_block = torch.cat([decoded_in_block, decoded_until_eos], dim=1)
                        current_ids = torch.cat([current_ids, decoded_in_block], dim=1)
                        decoded_token_total += int(decoded_until_eos.shape[1])
                        unresolved_len = 0
                        finished = True
                        break

                block_hidden, block_logits = _pack_block_tensors(denoise_hidden, denoise_logits, unresolved_len)
                diffusion_state = self._update_middle_state(
                    current_middle_state,
                    block_hidden,
                    block_logits,
                    meta["middle_slot_mask"],
                )

                decoded_in_block = torch.cat([decoded_in_block, decoded_ids], dim=1)
                decoded_token_total += int(decode_len)
                unresolved_len -= decode_len
                if unresolved_len <= 0:
                    break

                current_middle_state = _shift_middle_state(diffusion_state, decode_len, unresolved_len)

            if decoded_in_block.shape[1] > 0 and not finished:
                prefix_len = int(past_key_values.get_seq_length())
                current_prefix_logit, current_prefix_hidden, decode_attention_mask = _decode_tokens(
                    decoded_in_block,
                    past_key_values,
                    prefix_len,
                )
                cache_decode_forward_calls += 1
                if has_vision_inputs:
                    current_prefix_logit = self._mask_mm_forbidden_logits(current_prefix_logit)
                current_ids = torch.cat([current_ids, decoded_in_block], dim=1)
                current_attention_mask = decode_attention_mask

            if finished:
                break

        if not return_stats:
            return current_ids

        total_forward_calls = (
            prefill_forward_calls
            + diffusion_generate_calls
            + refine_generate_calls
            + cache_decode_forward_calls
        )
        stats = {
            "generated_tokens": int(current_ids.shape[1] - initial_len),
            "decoded_tokens": int(decoded_token_total),
            "prefill_forward_calls": int(prefill_forward_calls),
            "diffusion_generate_calls": int(diffusion_generate_calls),
            "refine_generate_calls": int(refine_generate_calls),
            "cache_decode_forward_calls": int(cache_decode_forward_calls),
            "total_forward_calls": int(total_forward_calls),
            "backbone_forward_calls": int(diffusion_generate_calls + refine_generate_calls),
            "tokens_per_forward": float(decoded_token_total) / float(max(1, diffusion_generate_calls)),
            "tokens_per_diffusion_generate": float(decoded_token_total) / float(max(1, diffusion_generate_calls)),
            "tokens_per_backbone_forward": float(decoded_token_total)
            / float(max(1, diffusion_generate_calls + refine_generate_calls + cache_decode_forward_calls)),
            "tokens_per_total_forward": float(decoded_token_total) / float(max(1, total_forward_calls)),
        }
        return current_ids, stats


MDVLModel = MDVLModelV6


__all__ = ["MDVLModel", "MDVLModelV6"]
