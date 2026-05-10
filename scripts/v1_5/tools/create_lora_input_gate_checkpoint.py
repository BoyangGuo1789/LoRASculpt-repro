#!/usr/bin/env python
"""Create an input-gated LoRA checkpoint from an existing target LoRA.

The gate is trained from allowed training prompts only. It learns a compact
bag-of-text-features classifier that predicts whether the target LoRA should be
active, then stores the gate as JSON next to the adapter weights.
"""

import argparse
import json
import math
import os
import random
import re
import shutil
from collections import Counter
from datetime import datetime, timezone


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--target-json", required=True)
    parser.add_argument("--source-json", action="append", required=True)
    parser.add_argument("--max-target", type=int, default=10000)
    parser.add_argument("--max-source", type=int, default=10000)
    parser.add_argument("--max-features", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--source-scale", type=float, default=0.0)
    parser.add_argument("--gate-projector", action="store_true")
    parser.add_argument("--no-gate-lora", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_checkpoint(src, dst, overwrite):
    if os.path.exists(dst):
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_json_or_jsonl(path):
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path) as f:
        return json.load(f)


def conversation_text(row):
    if isinstance(row, dict):
        if "text" in row:
            return row["text"]
        if "conversations" in row:
            for turn in row["conversations"]:
                if turn.get("from") == "human":
                    return turn.get("value", "")
        if "question" in row:
            return row["question"]
    return ""


def collect_texts(path, limit, seed):
    data = load_json_or_jsonl(path)
    rows = data if isinstance(data, list) else list(data.values())
    texts = [conversation_text(row) for row in rows]
    texts = [text for text in texts if text]
    rnd = random.Random(seed)
    rnd.shuffle(texts)
    return texts[:limit]


def features_from_text(text):
    text = (text or "").replace("<image>", " ")
    lower = text.lower()
    feats = Counter()
    for tok in re.findall(r"[a-z0-9]+", lower):
        feats[f"tok={tok}"] += 1
    option_count = len(re.findall(r"(?m)(?:^|\n)\s*[a-h]\.", lower))
    if option_count:
        feats["has_options"] = 1
        feats[f"option_count={min(option_count, 8)}"] = 1
    if "answer the question using a single word or phrase" in lower:
        feats["short_answer_instruction"] = 1
    if "short answer" in lower:
        feats["short_answer"] = 1
    return feats


def raw_score(feats, bias, weights):
    return bias + sum(weights.get(feat, 0.0) * value for feat, value in feats.items())


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def main():
    args = parse_args()
    target_texts = collect_texts(args.target_json, args.max_target, args.seed)
    per_source_limit = max(1, args.max_source // len(args.source_json))
    source_texts = []
    for idx, path in enumerate(args.source_json):
        source_texts.extend(collect_texts(path, per_source_limit, args.seed + idx + 1))
    source_texts = source_texts[: args.max_source]

    pos_counts = Counter()
    neg_counts = Counter()
    for text in target_texts:
        pos_counts.update(features_from_text(text))
    for text in source_texts:
        neg_counts.update(features_from_text(text))

    vocab = sorted(set(pos_counts) | set(neg_counts))
    total_pos = sum(pos_counts.values())
    total_neg = sum(neg_counts.values())
    vocab_size = max(1, len(vocab))
    scored = []
    for feat in vocab:
        p = (pos_counts[feat] + args.alpha) / (total_pos + args.alpha * vocab_size)
        n = (neg_counts[feat] + args.alpha) / (total_neg + args.alpha * vocab_size)
        weight = math.log(p / n)
        scored.append((abs(weight), feat, weight))
    scored.sort(reverse=True)
    weights = {feat: weight for _, feat, weight in scored[: args.max_features]}
    prior = math.log((len(target_texts) + args.alpha) / (len(source_texts) + args.alpha))

    pos_scores = [raw_score(features_from_text(text), prior, weights) for text in target_texts]
    neg_scores = [raw_score(features_from_text(text), prior, weights) for text in source_texts]
    pos_median = sorted(pos_scores)[len(pos_scores) // 2]
    neg_median = sorted(neg_scores)[len(neg_scores) // 2]
    threshold = 0.5 * (pos_median + neg_median)
    adjusted_bias = prior - threshold

    def gate(score):
        prob = sigmoid(score / max(args.temperature, 1e-6))
        return args.source_scale + (args.target_scale - args.source_scale) * prob

    pos_gate = [gate(score - threshold) for score in pos_scores]
    neg_gate = [gate(score - threshold) for score in neg_scores]

    copy_checkpoint(args.source_checkpoint, args.output_checkpoint, args.overwrite)
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "LoRA-IG",
        "description": "Input-conditioned gate applied inside PEFT LoRA Linear forward; no checkpoint routing.",
        "gate_type": "linear_text_bow",
        "source_checkpoint": args.source_checkpoint,
        "target_json": args.target_json,
        "source_json": args.source_json,
        "target_examples": len(target_texts),
        "source_examples": len(source_texts),
        "bias": adjusted_bias,
        "weights": weights,
        "temperature": args.temperature,
        "target_scale": args.target_scale,
        "source_scale": args.source_scale,
        "gate_lora": not bool(args.no_gate_lora),
        "gate_projector": bool(args.gate_projector),
        "default_gate": args.target_scale,
        "train_gate_stats": {
            "target_mean": sum(pos_gate) / len(pos_gate),
            "source_mean": sum(neg_gate) / len(neg_gate),
            "target_min": min(pos_gate),
            "source_max": max(neg_gate),
        },
        "note": "The gate is continuous LoRA-delta scaling from input text features; the adapter weights are unchanged.",
    }
    path = os.path.join(args.output_checkpoint, "lora_input_gate_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps({k: v for k, v in config.items() if k != "weights"}, indent=2))
    print(f"weights={len(weights)} written={path}")


if __name__ == "__main__":
    main()
