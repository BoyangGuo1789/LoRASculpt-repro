#!/usr/bin/env python
"""Build an OKVQA train split in LLaVA supervised fine-tuning format."""

import argparse
import collections
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-missing-images", action="store_true")
    return parser.parse_args()


def majority_answer(annotation):
    answers = [a.get("answer", "").strip() for a in annotation.get("answers", [])]
    answers = [a for a in answers if a]
    if not answers:
        return ""
    counts = collections.Counter(a.lower() for a in answers)
    best_norm = max(counts.items(), key=lambda kv: kv[1])[0]
    for answer in answers:
        if answer.lower() == best_norm:
            return answer
    return answers[0]


def main():
    args = parse_args()
    questions = json.load(open(args.questions))["questions"]
    annotations = json.load(open(args.annotations))["annotations"]
    ann_by_qid = {int(a["question_id"]): a for a in annotations}

    rows = []
    missing = 0
    for question in questions:
        qid = int(question["question_id"])
        ann = ann_by_qid.get(qid)
        if ann is None:
            continue
        answer = majority_answer(ann)
        if not answer:
            continue
        image = f"COCO_train2014_{int(question['image_id']):012d}.jpg"
        if args.skip_missing_images and args.image_root:
            if not os.path.exists(os.path.join(args.image_root, image)):
                missing += 1
                continue
        prompt = question["question"].strip()
        if not prompt.endswith("?"):
            prompt += "?"
        prompt += "\nAnswer the question using a single word or phrase."
        rows.append({
            "id": f"okvqa_train_{qid}",
            "image": image,
            "conversations": [
                {"from": "human", "value": f"<image>\n{prompt}"},
                {"from": "gpt", "value": answer},
            ],
            "source_task": "okvqa_train",
        })

    if args.sample_size and args.sample_size < len(rows):
        rng = random.Random(args.seed)
        rows = rng.sample(rows, args.sample_size)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2)

    print(json.dumps({
        "output": args.output,
        "num_rows": len(rows),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "missing_images": missing,
    }, indent=2))


if __name__ == "__main__":
    main()
