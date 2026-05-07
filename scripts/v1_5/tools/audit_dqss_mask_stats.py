#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import sys


def is_qkv(module_name):
    return any(proj in module_name for proj in ("q_proj", "k_proj", "v_proj"))


def in_dqss_scope(module_name, scope):
    if scope == "all":
        return True
    if scope == "none":
        return False
    tokens = [token.strip() for token in scope.replace("+", ",").split(",") if token.strip()]
    aliases = {
        "qkv": ("q_proj", "k_proj", "v_proj"),
        "qk": ("q_proj", "k_proj"),
        "qv": ("q_proj", "v_proj"),
        "kv": ("k_proj", "v_proj"),
        "q": ("q_proj",),
        "k": ("k_proj",),
        "v": ("v_proj",),
        "o": ("o_proj",),
        "mlp": ("gate_proj", "up_proj", "down_proj"),
    }
    if not tokens:
        tokens = ["qkv"]
    projections = []
    for token in tokens:
        projections.extend(aliases.get(token, (token,)))
    return any(proj in module_name for proj in projections)


def values(rows, key):
    return [float(row[key]) for row in rows if key in row and row[key] is not None]


def mean_or_nan(items):
    return statistics.mean(items) if items else float("nan")


def near(value, target, tol):
    return math.isfinite(value) and abs(value - target) <= tol


def main():
    parser = argparse.ArgumentParser(description="Audit LoRASculpt DQSS MIG-DIS mask stats.")
    parser.add_argument("path", help="Path to migdis_mask_stats.json")
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--rho-tol", type=float, default=0.03)
    parser.add_argument("--density-target", type=float, default=0.100005)
    parser.add_argument("--density-tol", type=float, default=0.002)
    parser.add_argument("--min-core-overlap", type=float, default=0.70)
    parser.add_argument("--max-aux-overlap", type=float, default=0.98)
    parser.add_argument("--require-mode", default="dqss")
    parser.add_argument("--dqss-scope", default="qkv")
    args = parser.parse_args()

    with open(args.path, "r") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list JSON in {args.path}, found {type(rows).__name__}")
    if not rows:
        raise SystemExit(f"No module rows found in {args.path}")

    qkv_rows = [row for row in rows if is_qkv(row.get("module", ""))]
    dqss_rows = [row for row in rows if in_dqss_scope(row.get("module", ""), args.dqss_scope)]
    non_dqss_rows = [row for row in rows if not in_dqss_scope(row.get("module", ""), args.dqss_scope)]
    non_qkv_rows = [row for row in rows if not is_qkv(row.get("module", ""))]
    failures = []

    bad_modes = sorted({row.get("selection_mode") for row in rows if row.get("selection_mode") != args.require_mode})
    if bad_modes:
        failures.append(f"selection_mode contains unexpected values: {bad_modes}")
    if not dqss_rows:
        failures.append(f"no module rows found for dqss scope {args.dqss_scope}")

    summary = {
        "modules_total": len(rows),
        "modules_qkv": len(qkv_rows),
        "modules_non_qkv": len(non_qkv_rows),
        "modules_dqss_scope": len(dqss_rows),
    }

    for side in ("A", "B"):
        density = mean_or_nan(values(rows, f"mask_density_{side}"))
        qkv_aux_frac = mean_or_nan(values(dqss_rows, f"dqss_aux_fraction_{side}"))
        qkv_core_overlap = mean_or_nan(values(dqss_rows, f"overlap_with_core_global_topk_{side}"))
        qkv_aux_overlap = mean_or_nan(values(dqss_rows, f"overlap_with_aux_global_topk_{side}"))
        qkv_guard_replacements = mean_or_nan(values(dqss_rows, f"dqss_guard_replacements_{side}"))
        non_qkv_aux_count = len(values(non_dqss_rows, f"dqss_aux_fraction_{side}"))

        summary[f"mask_density_{side}_mean"] = density
        summary[f"qkv_dqss_aux_fraction_{side}_mean"] = qkv_aux_frac
        summary[f"qkv_overlap_with_core_global_topk_{side}_mean"] = qkv_core_overlap
        summary[f"qkv_overlap_with_aux_global_topk_{side}_mean"] = qkv_aux_overlap
        summary[f"qkv_dqss_guard_replacements_{side}_mean"] = qkv_guard_replacements
        summary[f"non_scope_dqss_aux_fraction_{side}_count"] = non_qkv_aux_count

        if not near(density, args.density_target, args.density_tol):
            failures.append(
                f"mask_density_{side}_mean={density:.6f} outside "
                f"{args.density_target:.6f}+/-{args.density_tol:.6f}"
            )
        if not near(qkv_aux_frac, args.rho, args.rho_tol):
            failures.append(
                f"qkv_dqss_aux_fraction_{side}_mean={qkv_aux_frac:.6f} outside "
                f"{args.rho:.6f}+/-{args.rho_tol:.6f}"
            )
        if not math.isfinite(qkv_core_overlap) or qkv_core_overlap < args.min_core_overlap:
            failures.append(
                f"qkv_overlap_with_core_global_topk_{side}_mean={qkv_core_overlap:.6f} "
                f"< {args.min_core_overlap:.6f}"
            )
        if not math.isfinite(qkv_aux_overlap) or qkv_aux_overlap >= args.max_aux_overlap:
            failures.append(
                f"qkv_overlap_with_aux_global_topk_{side}_mean={qkv_aux_overlap:.6f} "
                f">= {args.max_aux_overlap:.6f}"
            )
        if non_qkv_aux_count:
            failures.append(f"non-scope rows unexpectedly have dqss_aux_fraction_{side}: {non_qkv_aux_count}")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        print("DQSS audit failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("DQSS audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
