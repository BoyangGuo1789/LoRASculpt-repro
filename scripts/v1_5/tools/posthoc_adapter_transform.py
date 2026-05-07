#!/usr/bin/env python
"""Create post-hoc LoRA adapter variants for LoRASculpt experiments.

The tool preserves the checkpoint format expected by the existing LLaVA/PEFT
loader. It can scale LoRA B matrices and optionally interpolate the saved
mm_projector trainables back toward the base model projector.
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


def scale_lora_b(checkpoint_dir, gamma):
    path = os.path.join(checkpoint_dir, "adapter_model.bin")
    state = torch.load(path, map_location="cpu")
    scaled = []
    for key, value in state.items():
        if key.endswith("lora_B.weight") and torch.is_floating_point(value):
            state[key] = (value.float() * gamma).to(dtype=value.dtype)
            scaled.append(key)
    torch.save(state, path)
    return scaled


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
    lora_keys = scale_lora_b(args.output_checkpoint, args.lora_gamma)
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
        "projector_beta": args.projector_beta,
        "base_mm_projector": args.base_mm_projector,
        "scaled_lora_b_tensors": len(lora_keys),
        "projector_tensors_interpolated": len(projector_keys),
        "note": "projector_beta=1 keeps target projector; 0 restores base projector; values in between shrink target projector delta.",
    }
    with open(os.path.join(args.output_checkpoint, "posthoc_adapter_transform_config.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
