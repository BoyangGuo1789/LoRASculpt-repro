#!/usr/bin/env python
"""Fuse LoRA adapters in dense-delta space, then project back to rank-r LoRA."""

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone

import torch


LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--basis_ckpts", required=True, help="Comma list like name=/path,name2=/path2")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--write_meta", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_basis(spec):
    out = {}
    for item in spec.split(","):
        if not item:
            continue
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def load_bin(path):
    return torch.load(path, map_location="cpu")


def load_adapter(path):
    model_path = os.path.join(path, "adapter_model.bin")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    state = load_bin(model_path)
    config_path = os.path.join(path, "adapter_config.json")
    config = json.load(open(config_path)) if os.path.exists(config_path) else {}
    rank = int(config.get("r") or 0)
    alpha = int(config.get("lora_alpha") or 0)
    return state, config, rank, alpha


def lora_modules(state):
    modules = {}
    extras = []
    for key in state:
        if key.endswith(".lora_A.weight"):
            modules.setdefault(key[:-len(".lora_A.weight")], {})["A"] = key
        elif key.endswith(".lora_B.weight"):
            modules.setdefault(key[:-len(".lora_B.weight")], {})["B"] = key
        else:
            extras.append(key)
    missing = [m for m, pair in modules.items() if "A" not in pair or "B" not in pair]
    if missing:
        raise ValueError(f"incomplete LoRA module pairs: {missing[:10]}")
    return modules, extras


def module_group(module):
    if ".q_proj" in module:
        return "q"
    if ".k_proj" in module:
        return "k"
    if ".v_proj" in module:
        return "v"
    if ".o_proj" in module:
        return "o"
    if any(x in module for x in [".gate_proj", ".up_proj", ".down_proj"]):
        return "mlp"
    return "other"


def layer_band(module):
    match = LAYER_RE.search(module)
    if not match:
        return "other"
    layer = int(match.group(1))
    if layer <= 7:
        return "early"
    if layer <= 23:
        return "mid"
    return "late"


def find_candidate(manifest, name):
    for candidate in manifest["candidates"]:
        if candidate["name"] == name:
            return candidate
    raise KeyError(f"candidate not found: {name}")


def coefficients_for_module(candidate, module):
    group = module_group(module)
    band = layer_band(module)
    coeffs = {"base": float(candidate.get("gamma_base", 0.90))}
    for rule in candidate.get("rules", []):
        if group not in rule.get("module_groups", []):
            continue
        if band not in rule.get("bands", []):
            continue
        basis = rule["basis"]
        lam = float(rule["lambda"])
        coeffs["base"] = coeffs.get("base", 0.0) - lam
        coeffs[basis] = coeffs.get(basis, 0.0) + lam
    return {k: v for k, v in coeffs.items() if abs(v) > 1e-12}


def compact_project(terms, rank, target_scaling):
    lefts = []
    rights = []
    for B, A, coeff, scaling in terms:
        signed = float(coeff) * float(scaling)
        if abs(signed) <= 1e-12:
            continue
        mag = math.sqrt(abs(signed))
        sign = 1.0 if signed >= 0 else -1.0
        lefts.append(B.float() * mag)
        rights.append(A.float() * (sign * mag))
    if not lefts:
        raise ValueError("no nonzero fusion terms")

    L = torch.cat(lefts, dim=1)
    R = torch.cat(rights, dim=0)
    ql, rl = torch.linalg.qr(L, mode="reduced")
    qr, rr = torch.linalg.qr(R.T, mode="reduced")
    k = rl @ rr.T
    u, s, vh = torch.linalg.svd(k, full_matrices=False)
    keep = min(rank, s.numel())
    denom = torch.sum(s * s).item()
    retained = (torch.sum(s[:keep] * s[:keep]).item() / denom) if denom else 1.0

    b_new = (ql @ u[:, :keep]) * (s[:keep] / target_scaling).unsqueeze(0)
    a_new = vh[:keep, :] @ qr.T
    if keep < rank:
        b_pad = torch.zeros(b_new.shape[0], rank - keep, dtype=b_new.dtype)
        a_pad = torch.zeros(rank - keep, a_new.shape[1], dtype=a_new.dtype)
        b_new = torch.cat([b_new, b_pad], dim=1)
        a_new = torch.cat([a_new, a_pad], dim=0)
    return b_new, a_new, retained, s.numel()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def main():
    args = parse_args()
    manifest = json.load(open(args.manifest))
    candidate = find_candidate(manifest, args.candidate)
    basis_paths = parse_basis(args.basis_ckpts)
    basis_paths["base"] = args.base_ckpt

    if os.path.exists(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(args.output_dir)
        shutil.rmtree(args.output_dir)
    shutil.copytree(args.base_ckpt, args.output_dir)

    states = {}
    configs = {}
    ranks = {}
    alphas = {}
    modules_by_basis = {}
    extras_seen = {}
    for name, path in basis_paths.items():
        state, config, rank, alpha = load_adapter(path)
        states[name] = state
        configs[name] = config
        ranks[name] = rank or args.rank
        alphas[name] = alpha or args.alpha
        modules, extras = lora_modules(state)
        modules_by_basis[name] = modules
        extras_seen[name] = extras

    base_modules = modules_by_basis["base"]
    base_set = set(base_modules)
    for name, modules in modules_by_basis.items():
        if set(modules) != base_set:
            raise ValueError(f"LoRA key mismatch for {name}: missing={sorted(base_set - set(modules))[:5]} extra={sorted(set(modules) - base_set)[:5]}")

    target_scaling = float(args.alpha) / float(args.rank)
    out_state = dict(states["base"])
    rows = []
    retained_values = []
    nan_inf = 0

    for module in sorted(base_modules):
        coeffs = coefficients_for_module(candidate, module)
        terms = []
        for name, coeff in coeffs.items():
            pair = modules_by_basis[name][module]
            A = states[name][pair["A"]]
            B = states[name][pair["B"]]
            scaling = float(alphas[name]) / float(ranks[name])
            terms.append((B, A, coeff, scaling))
        b_new, a_new, retained, total_rank = compact_project(terms, args.rank, target_scaling)
        pair = base_modules[module]
        b_dtype = states["base"][pair["B"]].dtype
        a_dtype = states["base"][pair["A"]].dtype
        out_state[pair["B"]] = b_new.to(dtype=b_dtype)
        out_state[pair["A"]] = a_new.to(dtype=a_dtype)
        if not torch.isfinite(b_new).all() or not torch.isfinite(a_new).all():
            nan_inf += 1
        retained_values.append(retained)
        rows.append({
            "module": module,
            "group": module_group(module),
            "band": layer_band(module),
            "coeffs": json.dumps(coeffs, sort_keys=True),
            "retained_energy": f"{retained:.8f}",
            "total_rank_before_projection": total_rank,
        })

    torch.save(out_state, os.path.join(args.output_dir, "adapter_model.bin"))

    config_path = os.path.join(args.output_dir, "adapter_config.json")
    if os.path.exists(config_path):
        config = json.load(open(config_path))
        config["r"] = args.rank
        config["lora_alpha"] = args.alpha
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    meta = {
        "candidate_name": args.candidate,
        "description": candidate.get("description", ""),
        "base_ckpt": args.base_ckpt,
        "output_dir": args.output_dir,
        "basis_ckpts": basis_paths,
        "gamma_base": candidate.get("gamma_base", manifest.get("default", {}).get("gamma_base")),
        "rules": candidate.get("rules", []),
        "rank": args.rank,
        "alpha": args.alpha,
        "target_scaling": target_scaling,
        "mask_mode": manifest.get("default", {}).get("mask_mode", "copy_base"),
        "num_lora_modules": len(base_modules),
        "num_mask_buffers_seen": sum(1 for items in extras_seen.values() for key in items if "mask" in key.lower()),
        "mean_retained_energy": sum(retained_values) / len(retained_values),
        "min_retained_energy": min(retained_values),
        "num_nan_or_inf_tensors": nan_inf,
        "source_git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.output_dir, "fusion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.output_dir, "fusion_manifest_used.json"), "w") as f:
        json.dump(candidate, f, indent=2)
    with open(os.path.join(args.output_dir, "fusion_per_module_stats.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
