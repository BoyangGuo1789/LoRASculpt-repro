import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.train.LoRASculpt_Trainer import LoRASculpt


class TPSAMIX(LoRASculpt):
    """Target-preserved SA-MIX training.

    This trainer is intentionally isolated from the baseline LoRASculpt path.
    It expects batches tagged by ``samix_is_source`` and optionally uses a
    frozen ``teacher`` LoRA adapter loaded by ``train.py``.
    """

    def _unwrap_model(self, model):
        return model.module if hasattr(model, "module") else model

    def _zero(self, model):
        return torch.zeros((), device=next(model.parameters()).device)

    def _normalize_lora_key(self, name: str) -> str:
        if name.startswith("module."):
            name = name[len("module."):]
        return name.replace(".default.weight", ".weight").replace(".teacher.weight", ".weight")

    def _base_model(self, model):
        core_model = self._unwrap_model(model)
        if hasattr(core_model, "get_base_model"):
            return core_model.get_base_model()
        return core_model

    def _image_patch_count(self, model) -> int:
        base_model = self._base_model(model)
        if hasattr(base_model, "get_vision_tower"):
            vision_tower = base_model.get_vision_tower()
            if hasattr(vision_tower, "num_patches"):
                return int(vision_tower.num_patches)
        return 576

    def _expand_labels_for_logits(self, model, inputs: Dict[str, Any], labels, logits):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        if input_ids is None:
            return labels

        target_len = logits.size(1)
        patch_count = self._image_patch_count(model)
        expanded_rows = []
        for row_ids, row_labels, row_mask in zip(input_ids, labels, attention_mask):
            keep = row_mask.bool() if row_mask is not None else torch.ones_like(row_ids, dtype=torch.bool)
            row_ids = row_ids[keep]
            row_labels = row_labels[keep]
            pieces = []
            for token_id, token_label in zip(row_ids, row_labels):
                if int(token_id.item()) == IMAGE_TOKEN_INDEX:
                    pieces.append(torch.full((patch_count,), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                else:
                    pieces.append(token_label.view(1))
            if pieces:
                row = torch.cat(pieces, dim=0)
            else:
                row = torch.empty((0,), device=labels.device, dtype=labels.dtype)
            row = row[:target_len]
            if row.numel() < target_len:
                pad = torch.full((target_len - row.numel(),), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)
                row = torch.cat([row, pad], dim=0)
            expanded_rows.append(row)
        return torch.stack(expanded_rows, dim=0)

    def _adapter_state_path(self, path: str) -> str:
        if os.path.isdir(path):
            return os.path.join(path, "adapter_model.bin")
        return path

    def _load_anchor_state(self, model):
        if hasattr(self, "_tp_anchor_state"):
            return self._tp_anchor_state

        path = getattr(self.args, "tp_samix_teacher_lora_path", None) or getattr(self.args, "lora_start_path", None)
        if not path:
            raise ValueError("TP-SA-MIX L2 anchor requires tp_samix_teacher_lora_path or lora_start_path")
        state_path = self._adapter_state_path(path)
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"TP-SA-MIX anchor adapter not found: {state_path}")

        raw_state = torch.load(state_path, map_location="cpu")
        self._tp_anchor_state = {self._normalize_lora_key(k): v.detach().float().cpu() for k, v in raw_state.items()}
        return self._tp_anchor_state

    def _lora_l2_anchor_loss(self, model):
        weight = float(getattr(self.args, "tp_samix_lambda_l2", 0.0) or 0.0)
        if weight <= 0.0:
            return self._zero(model)

        anchor = self._load_anchor_state(model)
        loss = self._zero(model)
        count = 0
        for name, param in self._unwrap_model(model).named_parameters():
            if "lora_" not in name or ".teacher." in name:
                continue
            key = self._normalize_lora_key(name)
            ref = anchor.get(key)
            if ref is None or tuple(ref.shape) != tuple(param.shape):
                continue
            diff = param.float() - ref.to(device=param.device, dtype=torch.float32)
            loss = loss + diff.pow(2).mean()
            count += 1
        if count == 0:
            raise ValueError("TP-SA-MIX L2 anchor found no matching LoRA parameters")
        return weight * loss / count

    def _weighted_ce_loss(self, logits, labels, is_source: Optional[torch.Tensor]):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)
        token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size).float(),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view_as(shift_labels)

        token_mask = shift_labels.ne(IGNORE_INDEX)
        sample_loss = (token_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        active = token_mask.any(dim=1)
        if is_source is None:
            weights = torch.ones_like(sample_loss)
        else:
            source_weight = float(getattr(self.args, "tp_samix_source_weight", 1.0) or 1.0)
            weights = torch.where(is_source.bool(), torch.full_like(sample_loss, source_weight), torch.ones_like(sample_loss))
        weights = weights * active.to(weights.dtype)
        return (sample_loss * weights).sum() / weights.sum().clamp_min(1.0)

    def _masked_ce_loss(self, logits, labels, sample_mask: Optional[torch.Tensor], weight: float = 1.0):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)
        token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size).float(),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view_as(shift_labels)

        token_mask = shift_labels.ne(IGNORE_INDEX)
        if sample_mask is not None:
            token_mask = token_mask & sample_mask[:, None].bool()
        if not token_mask.any():
            return logits.sum() * 0.0
        return float(weight) * token_loss[token_mask].mean()

    def _topk_kl_loss(self, student_logits, teacher_logits, labels, sample_mask: torch.Tensor, temperature: float, topk: int):
        shift_labels = labels[..., 1:].contiguous()
        token_mask = shift_labels.ne(IGNORE_INDEX) & sample_mask[:, None].bool()
        if not token_mask.any():
            return student_logits.sum() * 0.0

        student = student_logits[..., :-1, :][token_mask].float()
        teacher = teacher_logits[..., :-1, :][token_mask].float()
        gold = shift_labels[token_mask].unsqueeze(-1)

        k = max(1, min(int(topk), teacher.size(-1)))
        top_idx = teacher.topk(k, dim=-1).indices
        missing_gold = ~top_idx.eq(gold).any(dim=-1)
        if missing_gold.any():
            top_idx[missing_gold, -1:] = gold[missing_gold]

        student_sel = student.gather(1, top_idx)
        teacher_sel = teacher.gather(1, top_idx)
        t = max(float(temperature), 1e-6)
        student_log_prob = F.log_softmax(student_sel / t, dim=-1)
        teacher_prob = F.softmax(teacher_sel / t, dim=-1)
        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (t * t)

    def _teacher_logits(self, model, inputs: Dict[str, Any]):
        core_model = self._unwrap_model(model)
        if not hasattr(core_model, "set_adapter"):
            raise ValueError("TP-SA-MIX teacher KL requires a PEFT model with set_adapter")
        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        try:
            core_model.set_adapter("teacher")
            with torch.no_grad():
                return model(**teacher_inputs).logits.detach()
        finally:
            core_model.set_adapter("default")

    def _pcgrad_parameters(self, model):
        return [
            param
            for name, param in self._unwrap_model(model).named_parameters()
            if param.requires_grad and ".teacher." not in name
        ]

    def _pcgrad_surrogate_loss(self, model, target_loss, source_loss):
        params = self._pcgrad_parameters(model)
        if not params:
            raise ValueError("TP-SA-MIX PCGrad found no trainable parameters")

        target_grads = torch.autograd.grad(
            target_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        source_grads = torch.autograd.grad(
            source_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )

        dot = torch.zeros((), device=target_loss.device)
        target_norm_sq = torch.zeros((), device=target_loss.device)
        for target_grad, source_grad in zip(target_grads, source_grads):
            if target_grad is None or source_grad is None:
                continue
            dot = dot + (target_grad.float() * source_grad.float()).sum()
            target_norm_sq = target_norm_sq + target_grad.float().pow(2).sum()

        coeff = torch.where(
            dot < 0,
            dot / target_norm_sq.clamp_min(1e-12),
            torch.zeros_like(dot),
        )
        surrogate = torch.zeros((), device=target_loss.device)
        for param, target_grad, source_grad in zip(params, target_grads, source_grads):
            if target_grad is None and source_grad is None:
                continue
            if target_grad is None:
                combined = source_grad
            elif source_grad is None:
                combined = target_grad
            else:
                combined = target_grad + source_grad - coeff.to(source_grad.device) * target_grad
            surrogate = surrogate + (param.float() * combined.detach().float()).sum()
        return surrogate

    def _pcgrad_compute_loss(self, model, inputs, is_source, expanded_labels, outputs):
        source_weight = float(getattr(self.args, "tp_samix_source_weight", 1.0) or 1.0)
        target_mask = ~is_source.bool()
        source_mask = is_source.bool()

        target_loss = self._masked_ce_loss(outputs.logits, expanded_labels, target_mask, weight=1.0)
        source_loss = self._masked_ce_loss(outputs.logits, expanded_labels, source_mask, weight=source_weight)

        lambda_target_kl = float(getattr(self.args, "tp_samix_lambda_target_kl", 0.0) or 0.0)
        lambda_source_kl = float(getattr(self.args, "tp_samix_lambda_source_kl", 0.0) or 0.0)
        if lambda_target_kl > 0.0 or lambda_source_kl > 0.0:
            teacher_logits = self._teacher_logits(model, inputs)
            temperature = float(getattr(self.args, "tp_samix_kl_temperature", 2.0) or 2.0)
            topk = int(getattr(self.args, "tp_samix_kl_topk", 64) or 64)
            if lambda_target_kl > 0.0:
                target_loss = target_loss + lambda_target_kl * self._topk_kl_loss(
                    outputs.logits, teacher_logits, expanded_labels, target_mask, temperature, topk
                )
            if lambda_source_kl > 0.0:
                source_loss = source_loss + lambda_source_kl * self._topk_kl_loss(
                    outputs.logits, teacher_logits, expanded_labels, source_mask, temperature, topk
                )

        l2_loss = self._lora_l2_anchor_loss(model)
        actual_loss = target_loss + source_loss + l2_loss

        grad_loss = self._pcgrad_surrogate_loss(model, target_loss, source_loss) + l2_loss
        return actual_loss.detach() + (grad_loss - grad_loss.detach())

    def compute_loss(self, model, inputs, return_outputs=False):
        is_source = inputs.pop("samix_is_source", None)
        if is_source is not None:
            is_source = is_source.to(model.device) if hasattr(model, "device") else is_source

        labels = inputs["labels"]
        outputs = model(**inputs)
        expanded_labels = self._expand_labels_for_logits(model, inputs, labels, outputs.logits)
        if getattr(self.args, "tp_samix_use_pcgrad", False):
            if is_source is None:
                raise ValueError("TP-SA-MIX PCGrad requires samix_is_source batch tags")
            loss = self._pcgrad_compute_loss(model, inputs, is_source, expanded_labels, outputs)
            return (loss, outputs) if return_outputs else loss

        loss = self._weighted_ce_loss(outputs.logits, expanded_labels, is_source)

        lambda_target_kl = float(getattr(self.args, "tp_samix_lambda_target_kl", 0.0) or 0.0)
        lambda_source_kl = float(getattr(self.args, "tp_samix_lambda_source_kl", 0.0) or 0.0)
        if is_source is not None and (lambda_target_kl > 0.0 or lambda_source_kl > 0.0):
            teacher_logits = self._teacher_logits(model, inputs)
            temperature = float(getattr(self.args, "tp_samix_kl_temperature", 2.0) or 2.0)
            topk = int(getattr(self.args, "tp_samix_kl_topk", 64) or 64)
            target_mask = ~is_source.bool()
            source_mask = is_source.bool()
            if lambda_target_kl > 0.0:
                loss = loss + lambda_target_kl * self._topk_kl_loss(
                    outputs.logits, teacher_logits, expanded_labels, target_mask, temperature, topk
                )
            if lambda_source_kl > 0.0:
                loss = loss + lambda_source_kl * self._topk_kl_loss(
                    outputs.logits, teacher_logits, expanded_labels, source_mask, temperature, topk
                )

        loss = loss + self._lora_l2_anchor_loss(model)
        return (loss, outputs) if return_outputs else loss
