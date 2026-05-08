# coding=utf-8
"""Module-wise static LoRA gate calibration for LoRASculpt.

The trainer starts from a trained target LoRA, freezes the base model and LoRA
weights, and learns one scalar residual gate per LoRA module on a mixed
target/source calibration set. Before saving, the learned gates are baked into
the LoRA B matrices so evaluation still uses a normal static LoRA checkpoint.
"""

import json
import math
import os

import torch
from torch import nn
from transformers.trainer import Trainer

from llava.train.llava_trainer import LLaVATrainer


def _bool_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).lower() in ("1", "true", "yes", "y", "on")


class LoRASculptGateCal(LLaVATrainer):
    def _gatecal_float(self, name, default):
        return float(os.environ.get(name, default))

    def _gatecal_init(self):
        return min(max(self._gatecal_float("GATECAL_INIT", 0.95), 1e-4), 1.0 - 1e-4)

    def _gatecal_anchor(self):
        return min(max(self._gatecal_float("GATECAL_ANCHOR", 1.0), 0.0), 1.0)

    def _gatecal_reg_lambda(self):
        return self._gatecal_float("GATECAL_REG_LAMBDA", 1e-3)

    def _gatecal_log(self, message):
        if self.args.local_rank in (-1, 0):
            print(message, flush=True)

    def _gatecal_lora_b_modules(self):
        for name, module in self.model.named_modules():
            lora_b = getattr(module, "lora_B", None)
            if lora_b is None or "default" not in lora_b:
                continue
            yield name, lora_b["default"]

    def _gatecal_setup_once(self):
        if getattr(self, "gatecal_ready", False):
            return

        for param in self.model.parameters():
            param.requires_grad = False

        init = self._gatecal_init()
        init_logit = math.log(init / (1.0 - init))
        self.gatecal_modules = []
        self.gatecal_hooks = []

        for name, lora_b in self._gatecal_lora_b_modules():
            gate = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32, device=lora_b.weight.device))
            lora_b.register_parameter("gatecal_logit", gate)

            def _hook(module, _inputs, output):
                scale = torch.sigmoid(module.gatecal_logit).to(device=output.device, dtype=output.dtype)
                return output * scale

            self.gatecal_hooks.append(lora_b.register_forward_hook(_hook))
            self.gatecal_modules.append((name, lora_b))

        if not self.gatecal_modules:
            raise RuntimeError("LoRASculptGateCal found no LoRA B modules to calibrate")

        self.gatecal_ready = True
        self._gatecal_log(
            f"[GateCal] registered {len(self.gatecal_modules)} module gates; "
            f"init={init:.4f} anchor={self._gatecal_anchor():.4f} "
            f"reg_lambda={self._gatecal_reg_lambda():.2e}"
        )

    def create_optimizer(self):
        self._gatecal_setup_once()
        if self.optimizer is None:
            gate_params = []
            for _name, module in self.gatecal_modules:
                gate_params.append(module.gatecal_logit)

            if not gate_params:
                raise RuntimeError("LoRASculptGateCal has no trainable gate parameters")

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(
                [{"params": gate_params, "weight_decay": 0.0}],
                **optimizer_kwargs,
            )
            self._gatecal_log(f"[GateCal] optimizer will update {len(gate_params)} gate scalars")

        return self.optimizer

    def _gatecal_regularizer(self):
        if not getattr(self, "gatecal_ready", False):
            return None
        reg_lambda = self._gatecal_reg_lambda()
        if reg_lambda <= 0:
            return None
        anchor = self._gatecal_anchor()
        terms = []
        for _name, module in self.gatecal_modules:
            gate = torch.sigmoid(module.gatecal_logit.float())
            terms.append((gate - anchor) ** 2)
        return reg_lambda * torch.stack(terms).mean()

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs.pop("samix_is_source", None)
        outputs = model(**inputs)
        loss = outputs["loss"]
        reg = self._gatecal_regularizer()
        if reg is not None:
            loss = loss + reg.to(device=loss.device, dtype=loss.dtype)
        return (loss, outputs) if return_outputs else loss

    def _gatecal_stats(self):
        stats = []
        for name, module in getattr(self, "gatecal_modules", []):
            stats.append(
                {
                    "module": name,
                    "gate": float(torch.sigmoid(module.gatecal_logit.detach().float()).cpu().item()),
                }
            )
        if not stats:
            return {
                "count": 0,
                "mean": None,
                "min": None,
                "max": None,
                "modules": [],
            }
        gates = [row["gate"] for row in stats]
        return {
            "count": len(gates),
            "mean": float(sum(gates) / len(gates)),
            "min": float(min(gates)),
            "max": float(max(gates)),
            "modules": stats,
        }

    def apply_gate_calibration_to_lora(self):
        self._gatecal_setup_once()
        stats = self._gatecal_stats()
        with torch.no_grad():
            for _name, module in self.gatecal_modules:
                gate = torch.sigmoid(module.gatecal_logit.detach().float()).to(
                    device=module.weight.device, dtype=module.weight.dtype
                )
                module.weight.mul_(gate)

        for hook in getattr(self, "gatecal_hooks", []):
            hook.remove()
        for _name, module in self.gatecal_modules:
            if "gatecal_logit" in module._parameters:
                del module._parameters["gatecal_logit"]

        if self.args.local_rank in (-1, 0):
            os.makedirs(self.args.output_dir, exist_ok=True)
            path = os.path.join(self.args.output_dir, "gatecal_stats.json")
            with open(path, "w") as f:
                json.dump(stats, f, indent=2)
            self._gatecal_log(
                f"[GateCal] baked gates into LoRA B; mean={stats['mean']:.6f} "
                f"min={stats['min']:.6f} max={stats['max']:.6f}; stats={path}"
            )
