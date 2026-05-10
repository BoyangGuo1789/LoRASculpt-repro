#!/usr/bin/env python
"""Fuse a source LoRA into a target LoRA through target-orthogonal subspaces.

The output remains a normal rank-r PEFT LoRA checkpoint. For each selected
module this script forms the target dense delta Dt and source dense delta Ds,
removes the component of Ds aligned with Dt, then compresses the resulting
low-rank sum back to rank r:

    Df = target_gamma * Dt + source_lambda * (Ds - proj_Dt(Ds))

The low-rank compression is done with QR + a small SVD, so it avoids materializing
4096x4096 or 11008x4096 dense matrices.
"""

import argparse
import json
import math
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone

import torch


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SCOPE_TO_PROJECTIONS = {
    "all": set(PROJECTIONS),
    "attn": {"q_proj", "k_proj", "v_proj", "o_proj"},
    "qkvo": {"q_proj", "k_proj", "v_proj", "o_proj"},
    "qv": {"q_proj", "v_proj"},
    "v": {"v_proj"},
    "vo": {"v_proj", "o_proj"},
    "mlp": {"gate_proj", "up_proj", "down_proj"},
}
LAYER_BANDS = {
    "all": None,
    "early": (0, 7),
    "mid": (8, 23),
    "late": (24, 31),
    "early_mid": (0, 23),
    "mid_late": (8, 31),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--scope", choices=sorted(SCOPE_TO_PROJECTIONS), default="qv")
    parser.add_argument("--layer-band", choices=sorted(LAYER_BANDS), default="mid")
    parser.add_argument("--layers", default="", help="Optional explicit layer list/ranges, e.g. 8-23,27")
    parser.add_argument("--target-gamma", type=float, default=1.0)
    parser.add_argument("--source-lambda", type=float, required=True)
    parser.add_argument("--compress-rank", type=int, default=0, help="0 keeps target adapter rank")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--no-orthogonalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    return parser.parse_args()


def load_config(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "adapter_config.json")) as f:
        return json.load(f)


def adapter_scale(config):
    return float(config["lora_alpha"]) / float(config["r"])


def parse_layers(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            out.update(range(int(left), int(right) + 1))
        else:
            out.add(int(part))
    return out


def layer_index(key):
    match = re.search(r"\.layers\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def projection_name(key):
    for name in PROJECTIONS:
        if f".{name}." in key:
            return name
    return None


def selected_module(key, scope, layer_band, explicit_layers):
    proj = projection_name(key)
    if proj not in SCOPE_TO_PROJECTIONS[scope]:
        return False
    idx = layer_index(key)
    if idx is None:
        return layer_band == "all" and explicit_layers is None
    if explicit_layers is not None:
        return idx in explicit_layers
    band = LAYER_BANDS[layer_band]
    if band is None:
        return True
    return band[0] <= idx <= band[1]


def lora_b_key(a_key):
    return a_key.replace(".lora_A.weight", ".lora_B.weight")


def lowrank_inner(b_left, a_left, b_right, a_right):
    return torch.trace((b_left.T @ b_right) @ (a_right @ a_left.T))


def compress_product(x_factor, y_factor, rank):
    # x_factor: [out, k], y_factor: [k, in]
    qx, rx = torch.linalg.qr(x_factor, mode="reduced")
    qy, ry = torch.linalg.qr(y_factor.T, mode="reduced")
    core = rx @ ry.T
    u_core, s_vals, vh_core = torch.linalg.svd(core, full_matrices=False)
    keep = min(rank, s_vals.numel())
    total_energy = torch.sum(s_vals.square()).item()
    kept_energy = torch.sum(s_vals[:keep].square()).item()
    retained = kept_energy / total_energy if total_energy > 0 else 1.0
    u = qx @ u_core[:, :keep]
    vh = vh_core[:keep, :] @ qy.T
    sqrt_s = torch.sqrt(torch.clamp(s_vals[:keep], min=0))
    b_new = u * sqrt_s.unsqueeze(0)
    a_new = sqrt_s.unsqueeze(1) * vh
    return a_new.contiguous(), b_new.contiguous(), retained


def copy_checkpoint(src, dst, overwrite):
    if os.path.exists(dst):
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def summarize(values):
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def main():
    args = parse_args()
    torch.set_num_threads(args.num_threads)

    target_cfg = load_config(args.target_checkpoint)
    source_cfg = load_config(args.source_checkpoint)
    target_rank = int(target_cfg["r"])
    source_rank = int(source_cfg["r"])
    output_rank = args.compress_rank or target_rank
    if output_rank > target_rank + source_rank:
        raise ValueError("compress rank cannot exceed the combined target/source rank")
    if int(target_cfg["r"]) != output_rank:
        raise ValueError("output checkpoint keeps target adapter_config rank; use compress_rank equal to target rank")

    target_scale = adapter_scale(target_cfg)
    source_scale = adapter_scale(source_cfg)
    output_scale = target_scale
    explicit_layers = parse_layers(args.layers)

    target_state_path = os.path.join(args.target_checkpoint, "adapter_model.bin")
    source_state_path = os.path.join(args.source_checkpoint, "adapter_model.bin")
    target_state = torch.load(target_state_path, map_location="cpu")
    source_state = torch.load(source_state_path, map_location="cpu")

    selected = []
    skipped = []
    for key in sorted(k for k in target_state if k.endswith(".lora_A.weight")):
        b_key = lora_b_key(key)
        if key not in source_state or b_key not in source_state:
            skipped.append(key)
            continue
        if selected_module(key, args.scope, args.layer_band, explicit_layers):
            selected.append((key, b_key))

    if args.stats_only:
        out_state = dict(target_state)
    else:
        copy_checkpoint(args.target_checkpoint, args.output_checkpoint, args.overwrite)
        out_state = torch.load(os.path.join(args.output_checkpoint, "adapter_model.bin"), map_location="cpu")

    cosines = []
    projection_coefficients = []
    source_orthogonal_ratios = []
    retained_energies = []
    by_projection = Counter()
    by_layer = Counter()

    for a_key, b_key in selected:
        a_t = target_state[a_key].float()
        b_t = target_state[b_key].float()
        a_s = source_state[a_key].float()
        b_s = source_state[b_key].float()

        inner_unscaled = lowrank_inner(b_s, a_s, b_t, a_t).item()
        target_norm_unscaled = lowrank_inner(b_t, a_t, b_t, a_t).item()
        source_norm_unscaled = lowrank_inner(b_s, a_s, b_s, a_s).item()
        inner = source_scale * target_scale * inner_unscaled
        target_norm = (target_scale ** 2) * target_norm_unscaled
        source_norm = (source_scale ** 2) * source_norm_unscaled
        coeff = 0.0
        if not args.no_orthogonalize:
            coeff = inner / (target_norm + args.eps)
        target_coef = args.target_gamma - args.source_lambda * coeff
        source_coef = args.source_lambda

        denom = math.sqrt(max(source_norm * target_norm, args.eps))
        cosines.append(inner / denom if denom > 0 else 0.0)
        projection_coefficients.append(coeff)
        orth_norm = source_norm + (coeff ** 2) * target_norm - 2.0 * coeff * inner
        source_orthogonal_ratios.append(math.sqrt(max(orth_norm, 0.0) / max(source_norm, args.eps)))

        x_factor = torch.cat([b_t, b_s], dim=1)
        y_factor = torch.cat(
            [
                (target_coef * target_scale / output_scale) * a_t,
                (source_coef * source_scale / output_scale) * a_s,
            ],
            dim=0,
        )
        a_new, b_new, retained = compress_product(x_factor, y_factor, output_rank)
        retained_energies.append(retained)

        out_state[a_key] = a_new.to(dtype=target_state[a_key].dtype)
        out_state[b_key] = b_new.to(dtype=target_state[b_key].dtype)
        by_projection[projection_name(a_key)] += 1
        idx = layer_index(a_key)
        if idx is not None:
            by_layer[str(idx)] += 1

    if not args.stats_only:
        torch.save(out_state, os.path.join(args.output_checkpoint, "adapter_model.bin"))

    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "LoRA-OSF",
        "description": "Target-orthogonal source LoRA delta fusion with rank-preserving QR/SVD compression.",
        "target_checkpoint": args.target_checkpoint,
        "source_checkpoint": args.source_checkpoint,
        "output_checkpoint": args.output_checkpoint,
        "scope": args.scope,
        "layer_band": args.layer_band,
        "layers": args.layers,
        "target_gamma": args.target_gamma,
        "source_lambda": args.source_lambda,
        "orthogonalize": not args.no_orthogonalize,
        "target_rank": target_rank,
        "source_rank": source_rank,
        "output_rank": output_rank,
        "target_scale": target_scale,
        "source_scale": source_scale,
        "selected_modules": len(selected),
        "skipped_missing_source_modules": len(skipped),
        "selected_by_projection": dict(sorted(by_projection.items())),
        "selected_by_layer": dict(sorted(by_layer.items(), key=lambda item: int(item[0]))),
        "target_source_cosine": summarize(cosines),
        "projection_coefficient": summarize(projection_coefficients),
        "source_orthogonal_ratio": summarize(source_orthogonal_ratios),
        "retained_energy_after_rank_compression": summarize(retained_energies),
        "note": "Single static LoRA adapter; no task labels, prompt gates, checkpoint routing, or external data.",
    }
    if not args.stats_only:
        with open(os.path.join(args.output_checkpoint, "orthogonal_lora_fusion_config.json"), "w") as f:
            json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
