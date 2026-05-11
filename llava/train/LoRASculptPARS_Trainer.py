import os

import torch
import torch.nn.functional as F

from llava.train.LoRASculptMIGDIS_Trainer import CMR_LAMBDA, LoRASculptMIGDIS


class LoRASculptPARS(LoRASculptMIGDIS):
    def _pars_bool(self, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).lower() in ("1", "true", "yes", "y", "on")

    def _pars_enabled(self):
        return self._pars_bool(getattr(self.args, "pars_enable", os.environ.get("PARS_ENABLE", "false")))

    def _pars_arg(self, arg_name, env_name, default, cast):
        value = getattr(self.args, arg_name, None)
        if value is None:
            value = os.environ.get(env_name, default)
        return cast(value)

    def _pars_float(self, arg_name, env_name, default):
        return self._pars_arg(arg_name, env_name, default, float)

    def _pars_int(self, arg_name, env_name, default):
        return self._pars_arg(arg_name, env_name, default, int)

    def _pars_unwrap(self, model):
        if hasattr(model, "module"):
            model = model.module
        try:
            model = self.accelerator.unwrap_model(model)
        except Exception:
            pass
        return model

    def _pars_model_device(self, model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            return self.args.device

    def _pars_projector(self, model):
        model = self._pars_unwrap(model)
        core = model.get_model() if hasattr(model, "get_model") else getattr(model, "model", model)
        return getattr(core, "mm_projector", None)

    def _pars_init_projector_ref(self, model):
        if getattr(self, "_pars_projector_ref_ready", False):
            return
        projector = self._pars_projector(model)
        if projector is None:
            self._pars_projector_ref = {}
            self._pars_projector_ref_ready = True
            self._migdis_log("PARS projector_ref skipped: no mm_projector found")
            return
        self._pars_projector_ref = {
            name: param.detach().float().clone().cpu()
            for name, param in projector.named_parameters()
            if param.requires_grad
        }
        self._pars_projector_ref_ready = True
        self._migdis_log(f"PARS projector_ref initialized for {len(self._pars_projector_ref)} tensors")

    def _pars_projector_anchor_loss(self, model):
        self._pars_init_projector_ref(model)
        projector = self._pars_projector(model)
        if projector is None or not self._pars_projector_ref:
            zero = torch.zeros((), device=self._pars_model_device(model))
            return zero, 0.0, 0.0

        eps = self._pars_float("pars_eps", "PARS_EPS", 1e-8)
        device = self._pars_model_device(projector)
        delta_sq = torch.zeros((), device=device, dtype=torch.float32)
        ref_sq = torch.zeros((), device=device, dtype=torch.float32)
        for name, param in projector.named_parameters():
            if name not in self._pars_projector_ref:
                continue
            ref = self._pars_projector_ref[name].to(device=param.device, dtype=torch.float32)
            cur = param.float()
            delta_sq = delta_sq + (cur - ref).pow(2).sum()
            ref_sq = ref_sq + ref.pow(2).sum()

        drift = torch.sqrt(delta_sq / (ref_sq + eps) + eps)
        tau = self._pars_float("pars_projector_tau", "PARS_PROJECTOR_TAU", 0.020)
        lam = self._pars_float("pars_projector_lambda", "PARS_PROJECTOR_LAMBDA", 20.0)
        warmup = max(1, self._pars_int("pars_projector_warmup_steps", "PARS_PROJECTOR_WARMUP_STEPS", 100))
        ramp = min(1.0, float(self.state.global_step + 1) / float(warmup))
        excess = F.relu(drift - tau)
        loss = lam * ramp * excess.pow(2)
        return loss, float(drift.detach().cpu()), float(excess.detach().cpu())

    def _pars_lora_pairs(self, model):
        pairs = {}
        for name, param in model.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                continue
            key = name
            if key.startswith("module."):
                key = key[len("module."):]
            key = key.replace(".lora_A.default.weight", "")
            key = key.replace(".lora_B.default.weight", "")
            slot = pairs.setdefault(key, {})
            if "lora_A" in name:
                slot["A"] = param
            elif "lora_B" in name:
                slot["B"] = param
        return pairs

    def _pars_lora_orth_loss(self, model):
        lam = self._pars_float("pars_orth_lambda", "PARS_ORTH_LAMBDA", 0.01)
        if lam <= 0:
            return torch.zeros((), device=self._pars_model_device(model)), 0
        r_stable = self._pars_int("pars_stable_rank", "PARS_STABLE_RANK", 8)
        eps = self._pars_float("pars_eps", "PARS_EPS", 1e-8)
        losses = []
        for pair in self._pars_lora_pairs(model).values():
            A = pair.get("A")
            B = pair.get("B")
            if A is None or B is None or A.ndim != 2 or B.ndim != 2:
                continue
            rs = min(r_stable, A.shape[0] - 1, B.shape[1] - 1)
            if rs <= 0 or rs >= A.shape[0] or rs >= B.shape[1]:
                continue
            A_s = F.normalize(A[:rs].float(), dim=1, eps=eps)
            A_p = F.normalize(A[rs:].float(), dim=1, eps=eps)
            B_s = F.normalize(B[:, :rs].float(), dim=0, eps=eps)
            B_p = F.normalize(B[:, rs:].float(), dim=0, eps=eps)
            losses.append((A_s @ A_p.T).pow(2).mean() + (B_s.T @ B_p).pow(2).mean())
        if not losses:
            return torch.zeros((), device=self._pars_model_device(model)), 0
        return lam * torch.stack(losses).mean(), len(losses)

    def _pars_make_grad_scale_hook(self, name):
        def hook(grad):
            if grad is None or not self._pars_enabled() or grad.ndim != 2:
                return grad
            r_stable = self._pars_int("pars_stable_rank", "PARS_STABLE_RANK", 8)
            mult = self._pars_float("pars_stable_lr_mult", "PARS_STABLE_LR_MULT", 0.25)
            if mult == 1.0 or r_stable <= 0:
                return grad
            scaled = grad.clone()
            if "lora_A" in name:
                rs = min(r_stable, scaled.shape[0])
                scaled[:rs, :].mul_(mult)
            elif "lora_B" in name:
                rs = min(r_stable, scaled.shape[1])
                scaled[:, :rs].mul_(mult)
            else:
                return grad
            self._pars_grad_scale_updates = getattr(self, "_pars_grad_scale_updates", 0) + 1
            if not getattr(self, "_pars_grad_scale_logged", False):
                self._pars_grad_scale_logged = True
                self._migdis_log("PARS stable rank grad scaled via backward hook")
            return scaled
        return hook

    def _pars_register_grad_scale_hooks_once(self, model):
        if not self._pars_enabled() or getattr(self, "_pars_grad_scale_hooks_registered", False):
            return
        handles = []
        registered = 0
        for name, param in model.named_parameters():
            if not param.requires_grad or param.ndim != 2:
                continue
            if "lora_A" not in name and "lora_B" not in name:
                continue
            handles.append(param.register_hook(self._pars_make_grad_scale_hook(name)))
            registered += 1
        self._pars_grad_scale_hooks = handles
        self._pars_grad_scale_hooks_registered = True
        self._migdis_log(f"PARS stable rank grad scaling hooks registered for {registered} LoRA tensors")

    def _pars_apply_rank_grad_scaling(self, model):
        if not self._pars_enabled():
            return 0
        if getattr(self, "_pars_grad_scale_hooks_registered", False):
            return 0
        r_stable = self._pars_int("pars_stable_rank", "PARS_STABLE_RANK", 8)
        mult = self._pars_float("pars_stable_lr_mult", "PARS_STABLE_LR_MULT", 0.25)
        if mult == 1.0 or r_stable <= 0:
            return 0
        touched = 0
        for name, param in model.named_parameters():
            if param.grad is None or param.grad.ndim != 2:
                continue
            if "lora_A" in name:
                rs = min(r_stable, param.grad.shape[0])
                param.grad[:rs, :].mul_(mult)
                touched += 1
            elif "lora_B" in name:
                rs = min(r_stable, param.grad.shape[1])
                param.grad[:, :rs].mul_(mult)
                touched += 1
        return touched

    def _migdis_apply_masks_to_params_and_grads(self, model, apply_grads=False):
        super()._migdis_apply_masks_to_params_and_grads(model, apply_grads=apply_grads)
        if not apply_grads:
            return
        touched = self._pars_apply_rank_grad_scaling(model)
        log_every = max(1, self._pars_int("pars_log_every", "PARS_LOG_EVERY", 20))
        if touched and self.state.global_step % log_every == 0:
            self._migdis_log(f"PARS stable rank grad scaled on {touched} LoRA tensors")

    def training_step(self, model, inputs):
        self._pars_register_grad_scale_hooks_once(model)
        return super().training_step(model, inputs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        ce_loss = outputs["loss"]
        cmr_loss = self.comput_custom_reg(model, reg_lambda=CMR_LAMBDA)
        total = ce_loss + cmr_loss

        if self._pars_enabled():
            proj_loss, drift, excess = self._pars_projector_anchor_loss(model)
            orth_loss, orth_pairs = self._pars_lora_orth_loss(model)
            total = total + proj_loss + orth_loss
            log_every = max(1, self._pars_int("pars_log_every", "PARS_LOG_EVERY", 20))
            if self.state.global_step % log_every == 0:
                self._migdis_log(
                    f"PARS step={self.state.global_step} "
                    f"ce={float(ce_loss.detach().cpu()):.6f} "
                    f"cmr={float(cmr_loss.detach().cpu()):.6f} "
                    f"proj={float(proj_loss.detach().cpu()):.6f} "
                    f"orth={float(orth_loss.detach().cpu()):.6f} "
                    f"drift={drift:.6f} excess={excess:.6f} orth_pairs={orth_pairs}"
                )
        return (total, outputs) if return_outputs else total
