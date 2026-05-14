#!/usr/bin/env python
"""Create a rank-expanded residual LoRA adapter.

The output is one ordinary PEFT LoRA checkpoint. It preserves the target LoRA
subspace exactly in the first rank block and appends a scaled source/general
residual subspace in the second rank block:

    Delta_out = target_gamma * Delta_target + source_lambda * Delta_source

Modules outside the selected scope keep the target block and receive a zero
residual block, so inference still uses one always-on adapter without task
routing.
"""

import argparse
import json
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
    parser.add_argument("--output-rank", type=int, default=0, help="0 means target rank + source rank")
    parser.add_argument("--output-alpha", type=int, default=0, help="0 preserves target scaling by alpha/r")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-threads", type=int, default=8)
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_config(checkpoint_dir):
    return load_json(os.path.join(checkpoint_dir, "adapter_config.json"))


def load_state(checkpoint_dir):
    path = os.path.join(checkpoint_dir, "adapter_model.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def scaling(config):
    return float(config["lora_alpha"]) / float(config["r"])


def lora_b_key(a_key):
    return a_key.replace(".lora_A.weight", ".lora_B.weight")


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


def copy_checkpoint(src, dst, overwrite):
    if os.path.exists(dst):
        if not overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def lora_a_keys(state):
    return sorted(k for k in state if k.endswith(".lora_A.weight"))


def expand_pair(a_t, b_t, a_s, b_s, target_coef, source_coef, output_rank):
    target_rank = a_t.shape[0]
    residual_rank = output_rank - target_rank
    if residual_rank < 0:
        raise ValueError("output rank must be at least target rank")

    a_dtype = a_t.dtype
    b_dtype = b_t.dtype
    if residual_rank == 0:
        return (target_coef * a_t.float()).to(a_dtype), b_t

    if a_s is None or b_s is None:
        a_res = torch.zeros(residual_rank, a_t.shape[1], dtype=torch.float32)
        b_res = torch.zeros(b_t.shape[0], residual_rank, dtype=torch.float32)
    else:
        keep = min(residual_rank, a_s.shape[0], b_s.shape[1])
        a_res = torch.zeros(residual_rank, a_t.shape[1], dtype=torch.float32)
        b_res = torch.zeros(b_t.shape[0], residual_rank, dtype=torch.float32)
        a_res[:keep, :] = a_s[:keep, :].float()
        b_res[:, :keep] = source_coef * b_s[:, :keep].float()

    a_target = (target_coef * a_t.float()).to(torch.float32)
    b_target = b_t.float()
    a_out = torch.cat([a_target, a_res], dim=0).to(a_dtype)
    b_out = torch.cat([b_target, b_res], dim=1).to(b_dtype)
    return a_out.contiguous(), b_out.contiguous()


def main():
    args = parse_args()
    torch.set_num_threads(args.num_threads)

    target_cfg = load_config(args.target_checkpoint)
    source_cfg = load_config(args.source_checkpoint)
    target_state = load_state(args.target_checkpoint)
    source_state = load_state(args.source_checkpoint)

    target_rank = int(target_cfg["r"])
    source_rank = int(source_cfg["r"])
    output_rank = args.output_rank or (target_rank + source_rank)
    if output_rank < target_rank:
        raise ValueError("output rank must be at least target rank")

    target_scale = scaling(target_cfg)
    source_scale = scaling(source_cfg)
    output_scale = target_scale
    output_alpha = args.output_alpha or int(round(output_rank * output_scale))
    output_scale = float(output_alpha) / float(output_rank)
    target_coef = args.target_gamma * target_scale / output_scale
    source_coef = args.source_lambda * source_scale / output_scale

    explicit_layers = parse_layers(args.layers)
    copy_checkpoint(args.target_checkpoint, args.output_checkpoint, args.overwrite)
    out_state = dict(target_state)

    selected = []
    expanded = []
    missing_source = []
    by_projection = Counter()
    by_layer = Counter()

    for a_key in lora_a_keys(target_state):
        b_key = lora_b_key(a_key)
        if b_key not in target_state:
            raise ValueError(f"missing target B for {a_key}")

        use_source = selected_module(a_key, args.scope, args.layer_band, explicit_layers)
        a_s = source_state.get(a_key) if use_source else None
        b_s = source_state.get(b_key) if use_source else None
        if use_source:
            selected.append(a_key)
            if a_s is None or b_s is None:
                missing_source.append(a_key)
            by_projection[projection_name(a_key)] += 1
            idx = layer_index(a_key)
            if idx is not None:
                by_layer[str(idx)] += 1

        a_out, b_out = expand_pair(
            target_state[a_key],
            target_state[b_key],
            a_s,
            b_s,
            target_coef,
            source_coef if use_source else 0.0,
            output_rank,
        )
        out_state[a_key] = a_out
        out_state[b_key] = b_out
        expanded.append(a_key)

    torch.save(out_state, os.path.join(args.output_checkpoint, "adapter_model.bin"))

    config_path = os.path.join(args.output_checkpoint, "adapter_config.json")
    out_cfg = load_json(config_path)
    out_cfg["r"] = output_rank
    out_cfg["lora_alpha"] = output_alpha
    with open(config_path, "w") as f:
        json.dump(out_cfg, f, indent=2)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Residual-Expanded LoRA",
        "target_checkpoint": args.target_checkpoint,
        "source_checkpoint": args.source_checkpoint,
        "output_checkpoint": args.output_checkpoint,
        "scope": args.scope,
        "layer_band": args.layer_band,
        "layers": args.layers,
        "target_rank": target_rank,
        "source_rank": source_rank,
        "output_rank": output_rank,
        "output_alpha": output_alpha,
        "target_gamma": args.target_gamma,
        "source_lambda": args.source_lambda,
        "target_scale": target_scale,
        "source_scale": source_scale,
        "output_scale": output_scale,
        "target_coef_applied_to_A": target_coef,
        "source_coef_applied_to_B": source_coef,
        "num_modules_expanded": len(expanded),
        "num_modules_with_source_residual": len(selected) - len(missing_source),
        "num_missing_source_modules": len(missing_source),
        "selected_by_projection": dict(by_projection),
        "selected_by_layer": dict(by_layer),
        "description": "Single always-on rank-expanded adapter; first block preserves target LoRA, second block adds scaled source residual. No task gate or checkpoint routing.",
    }
    with open(os.path.join(args.output_checkpoint, "residual_expanded_lora_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
