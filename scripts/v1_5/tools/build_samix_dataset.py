#!/usr/bin/env python3
"""Build deterministic IconQA + COCO source-anchor mixed training JSON."""

import argparse
import copy
import json
import os
import random
from collections import Counter


def _load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list JSON: {path}")
    return data


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


def _normalize_iconqa_image(path):
    path = path.lstrip("/")
    if path.startswith("iconqa/iconqa_data/"):
        return path
    if path.startswith("iconqa/"):
        return os.path.join("iconqa", "iconqa_data", path).replace("\\", "/")
    return os.path.join("iconqa", "iconqa_data", path).replace("\\", "/")


def _normalize_coco_image(path):
    path = path.lstrip("/")
    if path.startswith("coco/"):
        return path
    return os.path.join("coco", path).replace("\\", "/")


def _normalize_sample(sample, source):
    out = copy.deepcopy(sample)
    if "image" not in out:
        raise KeyError(f"{source} sample missing image: {out.get('id')}")
    if "conversations" not in out:
        raise KeyError(f"{source} sample missing conversations: {out.get('id')}")
    if source == "iconqa":
        out["image"] = _normalize_iconqa_image(out["image"])
    elif source == "coco":
        out["image"] = _normalize_coco_image(out["image"])
    else:
        raise ValueError(source)
    out["samix_source"] = source
    return out


def _sample_coco(coco, count, seed):
    if count < 0:
        raise ValueError("--coco-samples must be non-negative")
    if count > len(coco):
        raise ValueError(f"Requested {count} COCO samples, only {len(coco)} available")
    rng = random.Random(seed)
    indices = list(range(len(coco)))
    rng.shuffle(indices)
    return [coco[i] for i in indices[:count]]


def _image_exists(data_root, image_path):
    return os.path.exists(os.path.join(data_root, image_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iconqa-json", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--coco-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()

    iconqa_raw = _load_json(args.iconqa_json)
    coco_raw = _load_json(args.coco_json)
    coco_selected = _sample_coco(coco_raw, args.coco_samples, args.seed)

    iconqa = [_normalize_sample(x, "iconqa") for x in iconqa_raw]
    coco = [_normalize_sample(x, "coco") for x in coco_selected]
    mixed = iconqa + coco

    shuffle_seed = args.seed if args.shuffle_seed is None else args.shuffle_seed
    rng = random.Random(shuffle_seed)
    rng.shuffle(mixed)

    counts = Counter(x["samix_source"] for x in mixed)
    missing_examples = []
    if args.check_images:
        for item in mixed:
            if not _image_exists(args.data_root, item["image"]):
                missing_examples.append(item["image"])
                if len(missing_examples) >= 10:
                    break
        if missing_examples:
            raise FileNotFoundError(
                "Missing images under data root, first examples: "
                + ", ".join(missing_examples)
            )

    manifest = {
        "method": "SA-MIX",
        "description": "IconQA target training mixed with bounded COCO caption source-anchor samples.",
        "iconqa_json": args.iconqa_json,
        "coco_json": args.coco_json,
        "output_json": args.output_json,
        "data_root": args.data_root,
        "seed": args.seed,
        "shuffle_seed": shuffle_seed,
        "counts": dict(counts),
        "total": len(mixed),
        "image_root_for_training": args.data_root,
        "check_images": bool(args.check_images),
        "missing_examples": missing_examples,
    }

    _write_json(args.output_json, mixed)
    _write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
