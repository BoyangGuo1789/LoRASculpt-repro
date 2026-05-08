#!/usr/bin/env python
"""Merge base and LoRA answer files with a prompt-form adaptive gate.

The gate is intentionally task-name agnostic: it looks only at the sample
prompt and enables the target LoRA answer when the prompt exposes a structured
multiple-choice answer form. Open-ended prompts use the base answer.
"""

import argparse
import json
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timezone


OPTION_RE = re.compile(r"(?m)^\s*[A-D]\.")
OPEN_ENDED_SUFFIX = "Answer the question using a single word or phrase."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--lora-answers", required=True)
    parser.add_argument("--base-answers", required=True)
    parser.add_argument("--output-answers", required=True)
    parser.add_argument("--stats-output", default="")
    parser.add_argument("--fallback-to-lora", action="store_true")
    return parser.parse_args()


def load_jsonl_queues_by_id(path):
    rows = defaultdict(deque)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["question_id"]].append(row)
    return rows


def pop_answer(rows, qid, source_name):
    queue = rows.get(qid)
    if not queue:
        raise KeyError(f"Missing {source_name} answer for question_id={qid}")
    return dict(queue.popleft())


def iter_questions(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prompt_form_gate(text):
    """Return 1 for structured MCQA prompts, 0 for open-ended VQA prompts."""
    option_count = len(OPTION_RE.findall(text or ""))
    return int(option_count >= 2 and OPEN_ENDED_SUFFIX not in (text or ""))


def main():
    args = parse_args()
    lora_rows = load_jsonl_queues_by_id(args.lora_answers)
    base_rows = load_jsonl_queues_by_id(args.base_answers)
    os.makedirs(os.path.dirname(args.output_answers), exist_ok=True)

    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_file": args.question_file,
        "lora_answers": args.lora_answers,
        "base_answers": args.base_answers,
        "output_answers": args.output_answers,
        "rule": "gate=1 when prompt has at least two A-D option lines and is not the open-ended VQA suffix",
        "total": 0,
        "gate_on_lora": 0,
        "gate_off_base": 0,
        "fallback_lora": 0,
    }

    with open(args.output_answers, "w") as out:
        for question in iter_questions(args.question_file):
            qid = question["question_id"]
            gate = prompt_form_gate(question.get("text", ""))
            stats["total"] += 1

            if gate:
                row = pop_answer(lora_rows, qid, "lora")
                source = "lora"
                stats["gate_on_lora"] += 1
            else:
                source = "base"
                try:
                    row = pop_answer(base_rows, qid, "base")
                except KeyError:
                    if not args.fallback_to_lora:
                        raise
                    row = pop_answer(lora_rows, qid, "lora")
                    source = "lora_fallback"
                    stats["fallback_lora"] += 1
                stats["gate_off_base"] += 1

            metadata = dict(row.get("metadata") or {})
            metadata["prompt_form_gate"] = gate
            metadata["prompt_form_source"] = source
            row["metadata"] = metadata
            out.write(json.dumps(row) + "\n")

    if args.stats_output:
        os.makedirs(os.path.dirname(args.stats_output), exist_ok=True)
        with open(args.stats_output, "w") as f:
            json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
