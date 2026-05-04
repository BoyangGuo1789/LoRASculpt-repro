import argparse
import json
import re
from pathlib import Path

PAPER = {
    "OKVQA": 53.52,
    "OCRVQA": 59.50,
    "GQA": 57.63,
    "TextVQA": 53.76,
    "IconQA": 85.26,
    "Source Avg": 56.10,
    "Target Avg": 85.26,
    "Avg": 70.68,
}

ORDER = ["OKVQA", "OCRVQA", "GQA", "TextVQA", "IconQA"]


def parse_summary(path):
    text = Path(path).read_text(errors="ignore")
    found = {}
    for task, value in re.findall(r"Accuracy on ([A-Za-z0-9]+):\s*([0-9.]+)%", text):
        found[task] = float(value)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    actual = parse_summary(args.summary)
    rows = []
    for task in ORDER:
        paper = PAPER[task]
        value = actual.get(task)
        rows.append({"task": task, "paper": paper, "reproduced": value, "delta": None if value is None else round(value - paper, 2)})

    source_tasks = ["OKVQA", "OCRVQA", "GQA", "TextVQA"]
    if all(actual.get(t) is not None for t in source_tasks):
        actual["Source Avg"] = round(sum(actual[t] for t in source_tasks) / len(source_tasks), 2)
    if actual.get("IconQA") is not None:
        actual["Target Avg"] = actual["IconQA"]
    if actual.get("Source Avg") is not None and actual.get("Target Avg") is not None:
        actual["Avg"] = round((actual["Source Avg"] + actual["Target Avg"]) / 2, 2)
    for task in ["Source Avg", "Target Avg", "Avg"]:
        value = actual.get(task)
        rows.append({"task": task, "paper": PAPER[task], "reproduced": value, "delta": None if value is None else round(value - PAPER[task], 2)})

    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    lines = ["| Task | Paper | Reproduced | Delta |", "|---|---:|---:|---:|"]
    for row in rows:
        rep = "N/A" if row["reproduced"] is None else f"{row['reproduced']:.2f}"
        delta = "N/A" if row["delta"] is None else f"{row['delta']:+.2f}"
        lines.append(f"| {row['task']} | {row['paper']:.2f} | {rep} | {delta} |")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
