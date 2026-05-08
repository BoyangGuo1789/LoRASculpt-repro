#!/usr/bin/env python
"""Create post-hoc LoRA adapter variants for LoRASculpt experiments.

The tool preserves the checkpoint format expected by the existing LLaVA/PEFT
loader. It can scale LoRA B matrices and optionally interpolate the saved
mm_projector trainables back toward the base model projector.

For GateCal-TR experiments, it can also blend learned module-wise GateCal
scales back toward the baseline adapter:

    scale = lora_gamma * (1 - gatecal_alpha * (1 - learned_gate))

This keeps the final checkpoint as a plain static LoRA adapter.
"""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--lora-gamma", type=float, default=1.0)
    parser.add_argument("--gatecal-stats", default="")
    parser.add_argument("--gatecal-alpha", type=float, default=0.0)
    parser.add_argument("--projector-beta", type=float, default=1.0)
    parser.add_argument("--base-mm-projector", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_checkpoint(src, dst, overwrite):
    if os.path.exists(dst):
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_gatecal_scales(path, alpha):
    if not path:
        if alpha != 0.0:
            raise ValueError("--gatecal-stats is required when --gatecal-alpha is non-zero")
        return {}
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("--gatecal-alpha must be in [0, 1]")

    with open(path) as f:
        raw = json.load(f)
    modules = raw.get("modules", [])
    if not isinstance(modules, list):
        raise ValueError(f"{path} does not contain a GateCal modules list")

    scales = {}
    for item in modules:
        module = item["module"]
        gate = float(item["gate"])
        scales[module] = 1.0 - alpha * (1.0 - gate)
    return scales


def scale_lora_b(checkpoint_dir, gamma, gatecal_scales=None):
    gatecal_scales = gatecal_scales or {}
    path = os.path.join(checkpoint_dir, "adapter_model.bin")
    state = torch.load(path, map_location="cpu")
    scaled = []
    applied_scales = []
    missing_gatecal = []
    for key, value in state.items():
        if key.endswith("lora_B.weight") and torch.is_floating_point(value):
            module = key[: -len(".lora_B.weight")]
            gate_scale = gatecal_scales.get(module)
            if gatecal_scales and gate_scale is None:
                missing_gatecal.append(module)
                continue
            scale = gamma * (gate_scale if gate_scale is not None else 1.0)
            state[key] = (value.float() * scale).to(dtype=value.dtype)
            scaled.append(key)
            applied_scales.append(scale)
    if missing_gatecal:
        preview = ", ".join(missing_gatecal[:5])
        raise KeyError(f"GateCal stats missing {len(missing_gatecal)} LoRA modules, e.g. {preview}")
    torch.save(state, path)
    return scaled, applied_scales


def load_base_projector(path):
    raw = torch.load(path, map_location="cpu")
    return {
        "base_model.model.model.mm_projector.0.weight": raw["model.mm_projector.0.weight"],
        "base_model.model.model.mm_projector.0.bias": raw["model.mm_projector.0.bias"],
        "base_model.model.model.mm_projector.2.weight": raw["model.mm_projector.2.weight"],
        "base_model.model.model.mm_projector.2.bias": raw["model.mm_projector.2.bias"],
    }


def interpolate_projector(checkpoint_dir, beta, base_mm_projector):
    if beta == 1.0:
        return []
    if not base_mm_projector:
        raise ValueError("--base-mm-projector is required when --projector-beta != 1.0")

    path = os.path.join(checkpoint_dir, "non_lora_trainables.bin")
    state = torch.load(path, map_location="cpu")
    base = load_base_projector(base_mm_projector)
    changed = []
    for key, value in state.items():
        if key not in base:
            continue
        base_value = base[key].to(dtype=torch.float32)
        delta = value.to(dtype=torch.float32) - base_value
        state[key] = (base_value + beta * delta).to(dtype=value.dtype)
        changed.append(key)
    torch.save(state, path)
    return changed


def main():
    args = parse_args()
    copy_checkpoint(args.source_checkpoint, args.output_checkpoint, args.overwrite)
    gatecal_scales = load_gatecal_scales(args.gatecal_stats, args.gatecal_alpha)
    lora_keys, applied_lora_scales = scale_lora_b(
        args.output_checkpoint,
        args.lora_gamma,
        gatecal_scales,
    )
    projector_keys = interpolate_projector(
        args.output_checkpoint,
        args.projector_beta,
        args.base_mm_projector,
    )

    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": args.source_checkpoint,
        "output_checkpoint": args.output_checkpoint,
        "lora_gamma": args.lora_gamma,
        "gatecal_stats": args.gatecal_stats,
        "gatecal_alpha": args.gatecal_alpha,
        "projector_beta": args.projector_beta,
        "base_mm_projector": args.base_mm_projector,
        "scaled_lora_b_tensors": len(lora_keys),
        "lora_scale_mean": sum(applied_lora_scales) / len(applied_lora_scales)
        if applied_lora_scales
        else None,
        "lora_scale_min": min(applied_lora_scales) if applied_lora_scales else None,
        "lora_scale_max": max(applied_lora_scales) if applied_lora_scales else None,
        "projector_tensors_interpolated": len(projector_keys),
        "note": "GateCal-TR applies module-wise trust-region LoRA_B scaling; projector_beta=1 keeps target projector; 0 restores base projector; values in between shrink target projector delta.",
    }
    with open(os.path.join(args.output_checkpoint, "posthoc_adapter_transform_config.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
