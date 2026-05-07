#!/usr/bin/env python
"""Append MB-LDF smoke/fusion/eval metadata to a CSV ledger."""

import argparse
import csv
import json
import os
from datetime import datetime, timezone


FIELDS = [
    "run_id", "stage", "candidate_name", "checkpoint_path", "gamma_base",
    "v5_rules", "dqss_rules", "v4_rules", "mask_mode", "rank", "alpha",
    "num_lora_modules", "mean_retained_energy", "min_retained_energy",
    "IconQA", "OKVQA", "OCRVQA", "GQA", "TextVQA", "SourceAvg", "Avg",
    "Delta_vs_exact_baseline", "Delta_vs_current_best",
    "required_gqa_for_target", "promote_to_full_eval", "stop_reason",
    "fusion_log_path", "eval_log_path", "summary_path", "git_commit",
    "pushed_to_public", "created_at",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion_meta", default="")
    parser.add_argument("--metrics_json", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--fusion_log_path", default="")
    parser.add_argument("--eval_log_path", default="")
    parser.add_argument("--pushed_to_public", default="false")
    return parser.parse_args()


def rules_for(meta, basis):
    rules = [r for r in meta.get("rules", []) if r.get("basis") == basis]
    return json.dumps(rules, sort_keys=True)


def main():
    args = parse_args()
    meta = json.load(open(args.fusion_meta)) if args.fusion_meta else {}
    metrics = json.load(open(args.metrics_json)) if args.metrics_json else {}
    vals = metrics.get("metrics", {})

    exact = 70.05375
    current = 70.09625
    avg = vals.get("Avg")
    row = {k: "" for k in FIELDS}
    row.update({
        "run_id": f"{args.stage}-{meta.get('candidate_name', metrics.get('run_name', 'unknown'))}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "stage": args.stage,
        "candidate_name": meta.get("candidate_name", metrics.get("run_name", "")),
        "checkpoint_path": meta.get("output_dir", metrics.get("checkpoint", "")) or metrics.get("checkpoint", ""),
        "gamma_base": meta.get("gamma_base", ""),
        "v5_rules": rules_for(meta, "v5"),
        "dqss_rules": rules_for(meta, "dqss"),
        "v4_rules": rules_for(meta, "v4"),
        "mask_mode": meta.get("mask_mode", ""),
        "rank": meta.get("rank", ""),
        "alpha": meta.get("alpha", ""),
        "num_lora_modules": meta.get("num_lora_modules", ""),
        "mean_retained_energy": meta.get("mean_retained_energy", ""),
        "min_retained_energy": meta.get("min_retained_energy", ""),
        "IconQA": vals.get("IconQA", ""),
        "OKVQA": vals.get("OKVQA", ""),
        "OCRVQA": vals.get("OCRVQA", ""),
        "GQA": vals.get("GQA", ""),
        "TextVQA": vals.get("TextVQA", ""),
        "SourceAvg": vals.get("SourceAvg", ""),
        "Avg": avg if avg is not None else "",
        "Delta_vs_exact_baseline": (avg - exact) if avg is not None else "",
        "Delta_vs_current_best": (avg - current) if avg is not None else "",
        "required_gqa_for_target": metrics.get("required_gqa_for_target", ""),
        "promote_to_full_eval": metrics.get("promote_to_full_eval", ""),
        "stop_reason": metrics.get("stop_reason", ""),
        "fusion_log_path": args.fusion_log_path,
        "eval_log_path": args.eval_log_path or metrics.get("log_file", ""),
        "summary_path": args.summary,
        "git_commit": meta.get("source_git_commit", metrics.get("git_commit", "")),
        "pushed_to_public": args.pushed_to_public,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    write_header = not os.path.exists(args.results_csv) or os.path.getsize(args.results_csv) == 0
    with open(args.results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2, default=str))


if __name__ == "__main__":
    main()
