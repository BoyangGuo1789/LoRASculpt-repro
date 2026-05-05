#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path

SOURCE_ORDER = ["OKVQA", "OCRVQA", "GQA", "TextVQA"]

PATTERNS = {
    "IconQA": r"Accuracy on IconQA:\s*([0-9.]+)%",
    "COCO": r"CIDEr on COCO:\s*([0-9.]+)",
    "OKVQA": r"Accuracy on OKVQA:\s*([0-9.]+)%",
    "OCRVQA": r"Accuracy on OCRVQA:\s*([0-9.]+)%",
    "GQA": r"Accuracy on GQA:\s*([0-9.]+)%",
    "TextVQA": r"Accuracy on TextVQA:\s*([0-9.]+)%",
}


def latest_value(text, key):
    matches = re.findall(PATTERNS[key], text)
    return float(matches[-1]) if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--target", required=True, choices=["iconqa_txt", "coco"])
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    text = Path(args.summary_file).read_text() if Path(args.summary_file).exists() else ""
    target_key = "IconQA" if args.target == "iconqa_txt" else "COCO"
    values = {key: latest_value(text, key) for key in SOURCE_ORDER}
    values[target_key] = latest_value(text, target_key)

    source_vals = [values[k] for k in SOURCE_ORDER if values[k] is not None]
    source_avg = sum(source_vals) / len(source_vals) if len(source_vals) == len(SOURCE_ORDER) else None
    target_val = values[target_key]
    avg = (source_avg + target_val) / 2 if source_avg is not None and target_val is not None else None

    row = {
        "method": args.method,
        "target_dataset": args.target,
        "OKVQA": values["OKVQA"],
        "OCRVQA": values["OCRVQA"],
        "GQA": values["GQA"],
        "TextVQA": values["TextVQA"],
        "Source": source_avg,
        "Target": target_val,
        "Avg": avg,
        "summary_file": args.summary_file,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(row, indent=2))

    def fmt(x):
        return "" if x is None else f"{x:.2f}"

    md = [
        "| Block | Method | Target Dataset | OKVQA | OCRVQA | GQA | TextVQA | Source | Target | Avg |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Recommended-config methods | {args.method} | {args.target} | {fmt(row['OKVQA'])} | {fmt(row['OCRVQA'])} | {fmt(row['GQA'])} | {fmt(row['TextVQA'])} | {fmt(row['Source'])} | {fmt(row['Target'])} | {fmt(row['Avg'])} |",
        "",
    ]
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
