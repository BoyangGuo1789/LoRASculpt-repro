#!/usr/bin/env python
"""Split a VQA question file by the prompt-form adaptive residual gate."""

import argparse
import json
import os

from prompt_form_gate_merge_answers import prompt_form_gate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    gate_on_path = os.path.join(args.output_dir, "gate_on.jsonl")
    gate_off_path = os.path.join(args.output_dir, "gate_off.jsonl")
    stats_path = os.path.join(args.output_dir, "gate_split_stats.json")

    stats = {
        "question_file": args.question_file,
        "limit": args.limit,
        "total": 0,
        "gate_on_lora": 0,
        "gate_off_base": 0,
    }
    with open(gate_on_path, "w") as on_file, open(gate_off_path, "w") as off_file:
        with open(args.question_file) as src:
            for line in src:
                if not line.strip():
                    continue
                if args.limit and stats["total"] >= args.limit:
                    break
                row = json.loads(line)
                stats["total"] += 1
                if prompt_form_gate(row.get("text", "")):
                    on_file.write(json.dumps(row) + "\n")
                    stats["gate_on_lora"] += 1
                else:
                    off_file.write(json.dumps(row) + "\n")
                    stats["gate_off_base"] += 1

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
