"""Input-conditioned LoRA gating utilities.

The gate is intentionally lightweight: it is stored inside a LoRA checkpoint as
JSON and applied to PEFT LoRA Linear layers at inference. The model remains a
single base model plus one adapter; no checkpoint routing is used.
"""

import json
import math
import os
import re
import types
from collections import Counter

import torch
import torch.nn.functional as F
from peft.tuners.lora import Linear as PeftLoraLinear
from peft.tuners.lora import transpose


GATE_CONFIG_NAME = "lora_input_gate_config.json"


def load_lora_input_gate_config(model_path):
    path = os.path.join(model_path, GATE_CONFIG_NAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _features_from_text(text):
    text = (text or "").replace("<image>", " ")
    lower = text.lower()
    feats = Counter()
    for tok in re.findall(r"[a-z0-9]+", lower):
        feats[f"tok={tok}"] += 1
    option_count = len(re.findall(r"(?m)(?:^|\n)\s*[a-h]\.", lower))
    if option_count:
        feats["has_options"] = 1
        feats[f"option_count={min(option_count, 8)}"] = 1
    if "answer the question using a single word or phrase" in lower:
        feats["short_answer_instruction"] = 1
    if "short answer" in lower:
        feats["short_answer"] = 1
    return feats


def score_lora_input_gate(config, text):
    feats = _features_from_text(text)
    weights = config.get("weights", {})
    raw = float(config.get("bias", 0.0))
    for feat, value in feats.items():
        raw += float(weights.get(feat, 0.0)) * float(value)
    return raw


def gate_value_from_text(config, text):
    score = score_lora_input_gate(config, text)
    temperature = max(float(config.get("temperature", 1.0)), 1e-6)
    low = float(config.get("source_scale", 0.0))
    high = float(config.get("target_scale", 1.0))
    prob = 1.0 / (1.0 + math.exp(-score / temperature))
    return low + (high - low) * prob


def _gated_lora_linear_forward(self, x):
    previous_dtype = x.dtype
    active_adapter = self.active_adapter
    if active_adapter not in self.lora_A.keys():
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    if self.disable_adapters:
        if self.r[active_adapter] > 0 and self.merged:
            self.unmerge()
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    elif self.r[active_adapter] > 0 and not self.merged:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        lora_x = x.to(self.lora_A[active_adapter].weight.dtype)
        lora_delta = (
            self.lora_B[active_adapter](
                self.lora_A[active_adapter](self.lora_dropout[active_adapter](lora_x))
            )
            * self.scaling[active_adapter]
        )
        owner = getattr(self, "_lora_input_gate_owner", None)
        gate = getattr(owner, "_lora_input_gate_value", 1.0) if owner is not None else 1.0
        gate_tensor = torch.as_tensor(gate, device=lora_delta.device, dtype=lora_delta.dtype)
        result += lora_delta * gate_tensor
    else:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    return result.to(previous_dtype)


def collect_projector_base_state(model):
    base_state = {}
    for key, value in model.state_dict().items():
        if "mm_projector" in key and (key.endswith(".weight") or key.endswith(".bias")):
            base_state[key] = value.detach().cpu().clone()
    return base_state


def _gated_projector_forward(self, x):
    target = F.linear(x, self.weight, self.bias)
    base_weight = self._lora_input_gate_base_weight.to(device=x.device, dtype=self.weight.dtype)
    base_bias = None
    if getattr(self, "_lora_input_gate_base_bias", None) is not None:
        base_bias = self._lora_input_gate_base_bias.to(device=x.device, dtype=self.weight.dtype)
    base = F.linear(x, base_weight, base_bias)
    owner = getattr(self, "_lora_input_gate_owner", None)
    gate = getattr(owner, "_lora_input_gate_value", 1.0) if owner is not None else 1.0
    gate_tensor = torch.as_tensor(gate, device=target.device, dtype=target.dtype)
    return base + gate_tensor * (target - base)


def apply_projector_input_gate(model, base_projector_state):
    if not base_projector_state:
        return 0
    patched = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "mm_projector" not in name:
            continue
        matched_weight = None
        matched_bias = None
        for key, value in base_projector_state.items():
            if not key.endswith(".weight"):
                continue
            module_key = key[: -len(".weight")]
            if name.endswith(module_key):
                matched_weight = value
                bias_key = module_key + ".bias"
                matched_bias = base_projector_state.get(bias_key)
                break
        if matched_weight is None:
            continue
        module._lora_input_gate_owner = model
        module._lora_input_gate_base_weight = matched_weight
        module._lora_input_gate_base_bias = matched_bias
        module.forward = types.MethodType(_gated_projector_forward, module)
        patched += 1
    model._lora_input_gate_patched_projector_modules = patched
    return patched


def apply_lora_input_gate(model, config):
    model._lora_input_gate_config = config
    model._lora_input_gate_value = float(config.get("default_gate", config.get("target_scale", 1.0)))
    model._lora_input_gate_last = {}
    model._lora_input_gate_lora_enabled = bool(config.get("gate_lora", True))

    def set_lora_gate_from_text(self, text):
        score = score_lora_input_gate(self._lora_input_gate_config, text)
        gate = gate_value_from_text(self._lora_input_gate_config, text)
        self._lora_input_gate_value = float(gate)
        self._lora_input_gate_last = {"gate": float(gate), "score": float(score), "text": text}
        return float(gate), float(score)

    model.set_lora_gate_from_text = types.MethodType(set_lora_gate_from_text, model)

    patched = 0
    if model._lora_input_gate_lora_enabled:
        for module in model.modules():
            if isinstance(module, PeftLoraLinear):
                module._lora_input_gate_owner = model
                module.forward = types.MethodType(_gated_lora_linear_forward, module)
                patched += 1
    model._lora_input_gate_patched_modules = patched
    return patched
