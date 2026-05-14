import torch
import torch.nn.functional as F

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.train.LoRASculpt_Trainer import CMR_LAMBDA, LoRASculpt


class SANLoRA(LoRASculpt):
    """Source-Activation Nulling LoRA.

    The trainer keeps the LoRASculpt pruning and CMR path, but mixed source
    samples are used only to suppress LoRA delta activations. Source captions
    are not optimized as an answer-prediction objective.
    """

    def _unwrap_model(self, model):
        return model.module if hasattr(model, "module") else model

    def _zero(self, model):
        return torch.zeros((), device=next(model.parameters()).device)

    def _san_lambda(self):
        return float(getattr(self.args, "san_lambda", 0.05) or 0.0)

    def _san_warmup_steps(self):
        return max(0, int(getattr(self.args, "san_warmup_steps", 100) or 0))

    def _san_scope(self):
        return str(getattr(self.args, "san_scope", "qkv") or "qkv").lower()

    def _san_log_every(self):
        return int(getattr(self.args, "san_log_every", 20) or 0)

    def _san_eps(self):
        return float(getattr(self.args, "san_eps", 1e-8) or 1e-8)

    def _san_scope_match(self, name):
        scope = self._san_scope()
        if scope in ("all", "*"):
            return True
        if scope == "qkv":
            return any(x in name for x in ("q_proj", "k_proj", "v_proj"))
        if scope == "qv":
            return any(x in name for x in ("q_proj", "v_proj"))
        if scope == "vo":
            return any(x in name for x in ("v_proj", "o_proj"))
        if scope == "attn":
            return any(x in name for x in ("q_proj", "k_proj", "v_proj", "o_proj"))
        if scope == "mlp":
            return any(x in name for x in ("gate_proj", "up_proj", "down_proj"))
        if scope == "mm_projector":
            return "mm_projector" in name
        return any(part and part in name for part in scope.split(","))

    def _san_scaling(self, module):
        scaling = getattr(module, "scaling", 1.0)
        if isinstance(scaling, dict):
            scaling = scaling.get("default", 1.0)
        return float(scaling)

    def _san_default_lora_b(self, module):
        lora_b = getattr(module, "lora_B", None)
        if lora_b is None:
            return None
        if hasattr(lora_b, "__contains__") and "default" in lora_b:
            return lora_b["default"]
        if hasattr(lora_b, "default"):
            return lora_b.default
        return None

    def _san_make_hook(self, name, scaling):
        def hook(_module, _inputs, output):
            mask = getattr(self, "_san_source_mask", None)
            if mask is None:
                return
            if isinstance(output, tuple):
                output = output[0]
            if not torch.is_tensor(output) or output.dim() < 2:
                return
            if output.size(0) != mask.numel() or not mask.any():
                return
            source_mask = mask.bool()
            target_mask = ~source_mask
            if not target_mask.any():
                return
            source_delta = output[source_mask].float() * scaling
            if source_delta.numel() == 0:
                return
            target_delta = output[target_mask].detach().float() * scaling
            source_energy = source_delta.pow(2).mean()
            target_energy = target_delta.pow(2).mean().clamp_min(self._san_eps())
            loss = source_energy / target_energy
            self._san_step_losses.append(loss)
            self._san_step_modules.append(name)

        return hook

    def _san_register_hooks_once(self, model):
        if getattr(self, "_san_hooks_registered", False):
            return
        self._san_hooks = []
        for name, module in self._unwrap_model(model).named_modules():
            if not self._san_scope_match(name):
                continue
            lora_b = self._san_default_lora_b(module)
            if lora_b is None:
                continue
            handle = lora_b.register_forward_hook(self._san_make_hook(name, self._san_scaling(module)))
            self._san_hooks.append(handle)
        self._san_hooks_registered = True
        if self.is_local_process_zero():
            print(f"SAN-LoRA hooks registered: {len(self._san_hooks)} modules, scope={self._san_scope()}")

    def _san_weight(self):
        base = self._san_lambda()
        warmup = self._san_warmup_steps()
        if base <= 0.0:
            return 0.0
        if warmup <= 0:
            return base
        return base * min(1.0, float(self.state.global_step + 1) / float(warmup))

    def _base_model(self, model):
        core_model = self._unwrap_model(model)
        if hasattr(core_model, "get_base_model"):
            return core_model.get_base_model()
        return core_model

    def _image_patch_count(self, model):
        base_model = self._base_model(model)
        if hasattr(base_model, "get_vision_tower"):
            vision_tower = base_model.get_vision_tower()
            if hasattr(vision_tower, "num_patches"):
                return int(vision_tower.num_patches)
        return 576

    def _expand_labels_for_logits(self, model, inputs, labels, logits):
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
            row = torch.cat(pieces, dim=0) if pieces else torch.empty((0,), device=labels.device, dtype=labels.dtype)
            row = row[:target_len]
            if row.numel() < target_len:
                pad = torch.full((target_len - row.numel(),), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)
                row = torch.cat([row, pad], dim=0)
            expanded_rows.append(row)
        return torch.stack(expanded_rows, dim=0)

    def _masked_ce_loss(self, logits, labels, sample_mask):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_mask = shift_labels.ne(IGNORE_INDEX) & sample_mask[:, None].bool()
        if not token_mask.any():
            return logits.sum() * 0.0
        vocab_size = shift_logits.size(-1)
        token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size).float(),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view_as(shift_labels)
        return token_loss[token_mask].mean()

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        try:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            self.accelerator.backward(loss)
        finally:
            self._san_source_mask = None
            self._san_step_losses = []
            self._san_step_modules = []

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        self._san_register_hooks_once(model)
        is_source = inputs.pop("samix_is_source", None)
        if is_source is not None:
            is_source = is_source.to(inputs["input_ids"].device).bool()

        self._san_step_losses = []
        self._san_step_modules = []
        self._san_source_mask = is_source
        outputs = model(**inputs)

        if is_source is None:
            loss = outputs["loss"]
            target_loss = loss.detach()
        else:
            expanded_labels = self._expand_labels_for_logits(model, inputs, inputs["labels"], outputs.logits)
            target_mask = ~is_source
            loss = self._masked_ce_loss(outputs.logits, expanded_labels, target_mask)
            target_loss = loss.detach()

        reg_loss = self.comput_custom_reg(model, reg_lambda=CMR_LAMBDA)
        loss = loss + reg_loss

        raw_san = self._zero(model)
        if self._san_step_losses:
            raw_san = torch.stack(self._san_step_losses).mean()
            loss = loss + self._san_weight() * raw_san

        log_every = self._san_log_every()
        if log_every > 0 and self.state.global_step % log_every == 0 and self.is_local_process_zero():
            print(
                "SAN-LoRA "
                f"step={self.state.global_step} target_ce={float(target_loss.float().item()):.6f} "
                f"cmr={float(reg_loss.detach().float().item()):.6f} "
                f"san_ratio={float(raw_san.detach().float().item()):.6e} "
                f"san_weight={self._san_weight():.6f} hooked={len(self._san_step_losses)}"
            )

        return (loss, outputs) if return_outputs else loss
