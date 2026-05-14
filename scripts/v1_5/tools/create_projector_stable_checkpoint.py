#!/usr/bin/env python
"""Create a projector-stable LoRA checkpoint.

The output keeps the source checkpoint's LoRA adapter unchanged but replaces
or shrinks the saved non-LoRA mm_projector trainables toward the base LLaVA
projector. This is a static single-checkpoint probe for training-time projector
stability regularization.
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
    parser.add_argument("--base-projector", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--projector-alpha", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--match-target-dtype", action="store_true", default=True)
    return parser.parse_args()


def copy_checkpoint(src, dst, overwrite):
    if os.path.exists(dst):
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def projector_suffix(key):
    marker = "mm_projector"
    if marker not in key:
        return None
    return key[key.index(marker) + len(marker):]


def main():
    args = parse_args()
    non_lora_path = os.path.join(args.source_checkpoint, "non_lora_trainables.bin")
    if not os.path.isfile(non_lora_path):
        raise FileNotFoundError(non_lora_path)
    if not os.path.isfile(args.base_projector):
        raise FileNotFoundError(args.base_projector)

    copy_checkpoint(args.source_checkpoint, args.output_checkpoint, args.overwrite)

    target_state = torch.load(non_lora_path, map_location="cpu")
    base_projector = torch.load(args.base_projector, map_location="cpu")
    base_by_suffix = {projector_suffix(k): v for k, v in base_projector.items() if projector_suffix(k) is not None}

    replaced = []
    for key, target_tensor in list(target_state.items()):
        suffix = projector_suffix(key)
        if suffix is None:
            continue
        if suffix not in base_by_suffix:
            raise KeyError(f"No base projector tensor matching suffix {suffix!r} for target key {key}")
        base_tensor = base_by_suffix[suffix]
        if tuple(base_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"Shape mismatch for {key}: base={tuple(base_tensor.shape)} target={tuple(target_tensor.shape)}"
            )
        base_tensor = base_tensor.to(dtype=target_tensor.dtype) if args.match_target_dtype else base_tensor
        alpha = float(args.projector_alpha)
        new_tensor = base_tensor + alpha * (target_tensor - base_tensor)
        diff_l2 = torch.linalg.vector_norm(target_tensor.float() - new_tensor.float()).item()
        target_l2 = torch.linalg.vector_norm(target_tensor.float()).item()
        target_state[key] = new_tensor.clone()
        replaced.append(
            {
                "key": key,
                "shape": list(target_tensor.shape),
                "old_dtype": str(target_tensor.dtype),
                "new_dtype": str(new_tensor.dtype),
                "target_base_l2": diff_l2,
                "relative_l2": diff_l2 / max(target_l2, 1e-12),
            }
        )

    if not replaced:
        raise RuntimeError("No mm_projector tensors were found in non_lora_trainables.bin")

    output_non_lora_path = os.path.join(args.output_checkpoint, "non_lora_trainables.bin")
    torch.save(target_state, output_non_lora_path)
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "PSL",
        "description": "Projector-Stable LoRASculpt probe: keep target LoRA, shrink target mm_projector trainables toward the base projector.",
        "projector_alpha": float(args.projector_alpha),
        "source_checkpoint": args.source_checkpoint,
        "base_projector": args.base_projector,
        "output_checkpoint": args.output_checkpoint,
        "replaced_tensors": replaced,
        "note": "Single checkpoint; no input gating, no task routing, no LoRA deactivation. alpha=0 uses the base projector; alpha=1 recovers the source checkpoint projector.",
    }
    meta_path = os.path.join(args.output_checkpoint, "projector_stable_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
