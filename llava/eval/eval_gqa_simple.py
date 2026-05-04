import argparse
import json
import os
import string


def normalize_answer(text):
    text = str(text).strip().lower()
    return text.strip(string.punctuation + " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary-output-dir", default=None)
    args = parser.parse_args()

    with open(args.annotation_file) as f:
        annotations = json.load(f)
    results = [json.loads(line) for line in open(args.result_file)]

    total = 0
    correct = 0
    for result in results:
        qid = str(result["question_id"])
        if qid not in annotations:
            continue
        pred = normalize_answer(result.get("text", ""))
        gold = normalize_answer(annotations[qid].get("answer", ""))
        total += 1
        correct += int(pred == gold)

    acc = 100.0 * correct / total if total else 0.0
    msg = f"Samples: {total}\nAccuracy: {acc:.2f}%\n"
    print(msg)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "result-gqa.txt"), "w") as f:
            f.write(msg)
    if args.summary_output_dir is not None:
        with open(args.summary_output_dir, "a") as f:
            f.write(f"\nSamples: {total}\nAccuracy on GQA: {acc:.2f}%\n")


if __name__ == "__main__":
    main()
