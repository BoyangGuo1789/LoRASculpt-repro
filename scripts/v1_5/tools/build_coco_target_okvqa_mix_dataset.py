#!/usr/bin/env python3
"""Build a deterministic COCO-target + OKVQA source-anchor TFR dataset."""

import argparse
import copy
import json
import os
import random
from collections import Counter


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list JSON: {path}")
    return data


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


def normalize_coco_target_image(path):
    path = path.lstrip("/")
    if path.startswith("coco/"):
        return path
    if path.startswith("train2014/"):
        return os.path.join("coco", path).replace("\\", "/")
    return os.path.join("coco", "coco_2014", "train2014", path).replace("\\", "/")


def normalize_okvqa_image(path):
    path = path.lstrip("/")
    if path.startswith("coco/train2014/"):
        return path
    if path.startswith("train2014/"):
        return os.path.join("coco", path).replace("\\", "/")
    if path.startswith("coco/"):
        return path
    return os.path.join("coco", "train2014", path).replace("\\", "/")


def normalize_sample(sample, source):
    out = copy.deepcopy(sample)
    if "image" not in out:
        raise KeyError(f"{source} sample missing image: {out.get('id')}")
    if "conversations" not in out:
        raise KeyError(f"{source} sample missing conversations: {out.get('id')}")
    if source == "target":
        out["image"] = normalize_coco_target_image(out["image"])
    elif source == "okvqa":
        out["image"] = normalize_okvqa_image(out["image"])
    else:
        raise ValueError(source)
    out["samix_source"] = source
    return out


def maybe_sample(rows, sample_size, seed):
    if sample_size <= 0 or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-target-json", required=True)
    parser.add_argument("--okvqa-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--okvqa-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()

    target = [normalize_sample(x, "target") for x in load_json(args.coco_target_json)]
    okvqa = [normalize_sample(x, "okvqa") for x in load_json(args.okvqa_json)]
    okvqa = maybe_sample(okvqa, args.okvqa_samples, args.seed)

    mixed = target + okvqa
    shuffle_seed = args.seed if args.shuffle_seed is None else args.shuffle_seed
    random.Random(shuffle_seed).shuffle(mixed)

    missing = []
    if args.check_images:
        for item in mixed:
            image_path = os.path.join(args.data_root, item["image"])
            if not os.path.exists(image_path):
                missing.append(item["image"])
                if len(missing) >= 10:
                    break
        if missing:
            raise FileNotFoundError("Missing images under data root: " + ", ".join(missing))

    counts = Counter(x["samix_source"] for x in mixed)
    manifest = {
        "method": "COCO-target TFR-BS data mix",
        "description": "COCO-Caption target samples plus bounded OKVQA source-anchor samples.",
        "coco_target_json": args.coco_target_json,
        "okvqa_json": args.okvqa_json,
        "output_json": args.output_json,
        "data_root": args.data_root,
        "okvqa_samples": args.okvqa_samples,
        "seed": args.seed,
        "shuffle_seed": shuffle_seed,
        "counts": dict(counts),
        "total": len(mixed),
        "check_images": bool(args.check_images),
        "missing_examples": missing,
    }
    write_json(args.output_json, mixed)
    write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
