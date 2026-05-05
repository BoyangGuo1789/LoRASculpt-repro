#!/usr/bin/env python
"""Fuse a LoRA-adapted LLaVA checkpoint with AdaDARE-gamma selection.

This script is intentionally explicit about provenance: no official AdaDARE-gamma
code repository was found during the source audit, so this implements the paper
formula for adaptive DARE probabilities. It operates on the merged task delta:
    theta_fused = theta_pre + gamma * (M * delta / q)
where q_i = clamp(n * (1 - sparsity) * score_i / sum(score), max=1).
With no released Hessian code, the default score uses a unit-Hessian proxy:
    score_i = |delta_i|.
"""

import argparse
import json
import os
import random
import shutil
from datetime import datetime, timezone

import torch

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True, help="LoRA checkpoint trained on the target task.")
    parser.add_argument("--base-model", required=True, help="Pretrained LLaVA base checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory for the fused AdaDARE-gamma checkpoint.")
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hessian-mode", choices=["unit"], default="unit")
    parser.add_argument("--deterministic", action="store_true", help="Keep the highest-scoring deltas instead of Bernoulli sampling.")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--dry-run", action="store_true", help="Load models and report delta stats without saving.")
    return parser.parse_args()


def adadare_delta(delta, sparsity, generator, deterministic=False, eps=1e-8):
    if not torch.is_floating_point(delta):
        return torch.zeros_like(delta)

    delta_f = delta.float()
    score = delta_f.abs()
    numel = score.numel()
    if numel == 0:
        return torch.zeros_like(delta_f)

    score_sum = score.sum()
    if not torch.isfinite(score_sum) or score_sum.item() == 0.0:
        return torch.zeros_like(delta_f)

    retain_budget = max(1.0, float(numel) * (1.0 - sparsity))
    retain_prob = (retain_budget * score / score_sum).clamp(max=1.0)

    if deterministic:
        k = max(1, int(round(retain_budget)))
        flat_score = score.flatten()
        if k >= numel:
            mask = torch.ones_like(delta_f)
            retain_prob = torch.ones_like(delta_f)
        else:
            threshold = torch.topk(flat_score, k, largest=True).values[-1]
            mask = (score >= threshold).to(delta_f.dtype)
            # Deterministic top-k does not sample from q; use global DARE rescale.
            retain_prob = torch.full_like(delta_f, max(1.0 - sparsity, eps))
    else:
        mask = torch.bernoulli(retain_prob, generator=generator).to(delta_f.dtype)

    return delta_f * mask / retain_prob.clamp_min(eps)


def load_model(model_path, model_base):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        device_map="cpu",
        device="cpu",
    )
    model.eval()
    return tokenizer, model


def main():
    args = parse_args()
    if not 0.0 <= args.sparsity < 1.0:
        raise ValueError("--sparsity must be in [0, 1).")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    print(f"[AdaDARE-gamma] loading LoRA adapter from {args.adapter_path}")
    tokenizer, tuned_model = load_model(args.adapter_path, args.base_model)
    print(f"[AdaDARE-gamma] loading base model from {args.base_model}")
    _, base_model = load_model(args.base_model, None)

    tuned_params = dict(tuned_model.named_parameters())
    changed_tensors = 0
    total_tensors = 0
    total_elements = 0
    changed_elements = 0

    with torch.no_grad():
        for name, base_param in base_model.named_parameters():
            tuned_param = tuned_params.get(name)
            if tuned_param is None or tuned_param.shape != base_param.shape or not torch.is_floating_point(base_param):
                continue
            total_tensors += 1
            delta = tuned_param.detach().cpu().float() - base_param.detach().cpu().float()
            max_abs = delta.abs().max().item() if delta.numel() else 0.0
            if max_abs == 0.0:
                continue
            adapted_delta = adadare_delta(
                delta,
                sparsity=args.sparsity,
                generator=generator,
                deterministic=args.deterministic,
            )
            fused = base_param.detach().cpu().float() + args.gamma * adapted_delta
            base_param.data.copy_(fused.to(dtype=base_param.dtype))
            changed_tensors += 1
            total_elements += delta.numel()
            changed_elements += int((adapted_delta != 0).sum().item())
            print(f"[AdaDARE-gamma] fused {name}: shape={tuple(delta.shape)} delta_max={max_abs:.6g}")
            del delta, adapted_delta, fused

    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_path": args.adapter_path,
        "base_model": args.base_model,
        "gamma": args.gamma,
        "sparsity": args.sparsity,
        "seed": args.seed,
        "hessian_mode": args.hessian_mode,
        "deterministic": args.deterministic,
        "total_float_tensors_seen": total_tensors,
        "changed_tensors": changed_tensors,
        "changed_elements_after_selection": changed_elements,
        "total_changed_tensor_elements": total_elements,
        "observed_retention_ratio": (changed_elements / total_elements) if total_elements else 0.0,
        "implementation_note": "Self implementation from CVPR 2025 paper/supplement; no official AdaDARE-gamma GitHub code found in audit.",
    }
    print(json.dumps(stats, indent=2))

    if args.dry_run:
        return

    os.makedirs(args.output_dir, exist_ok=True)
    base_model.save_pretrained(args.output_dir, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "adadare_gamma_config.json"), "w") as f:
        json.dump(stats, f, indent=2)

    src_config = os.path.join(args.adapter_path, "config.json")
    if os.path.exists(src_config):
        shutil.copy2(src_config, os.path.join(args.output_dir, "source_adapter_config.json"))

    print(f"[AdaDARE-gamma] saved fused checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
