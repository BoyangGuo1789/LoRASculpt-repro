#!/bin/bash
set -euo pipefail

CHECKPOINT=""
RUN_NAME=""
TASKS="iconqa,okvqa,ocrvqa,textvqa"
OUTPUT_ROOT=""
LOG_FILE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

: "${ROOT:=/data/guoboyang/LoRa-Projects/LoRASculpt-repro}"
: "${MODEL_BASE:=$ROOT/models/llava-v1.5-7b-ft}"
: "${PYTHON_BIN:=/data/guoboyang/miniconda3/envs/lorasculpt/bin/python}"
: "${OFFICIAL_JSON_ROOT:=$ROOT/downloads/official_issue2/extracted/LoRASculpt_JSON_files}"
: "${OUTPUT_ROOT:=$ROOT/repro_results/mbldf_plus1}"

if [ -z "$CHECKPOINT" ] || [ -z "$RUN_NAME" ]; then
  echo "Usage: $0 --checkpoint CKPT --run-name RUN [--tasks iconqa,okvqa,ocrvqa,textvqa] [--output-root DIR] [--log-file FILE]" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/$RUN_NAME"
SUMMARY="$OUTPUT_ROOT/$RUN_NAME/summary.txt"
METRICS="$OUTPUT_ROOT/$RUN_NAME/metrics.json"
CKPT_NAME="llava-v1.5-7b"
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

if [ -n "$LOG_FILE" ]; then
  echo "[eval_selected] log_file=$LOG_FILE"
fi

echo "Model: $CHECKPOINT" > "$SUMMARY"
echo "Official JSON root: $OFFICIAL_JSON_ROOT" >> "$SUMMARY"
echo "" >> "$SUMMARY"

has_task() {
  case ",$TASKS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

run_vqa() {
  local split="$1"
  local question_file="$2"
  local image_folder="$3"
  local out_dir="$OUTPUT_ROOT/$RUN_NAME/$split/$CKPT_NAME"

  mkdir -p "$out_dir"
  rm -f "$out_dir"/"${CHUNKS}"_*.jsonl "$out_dir/answers.jsonl"
  echo "[$split] question_file=$question_file"
  echo "[$split] image_folder=$image_folder"

  for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} "$PYTHON_BIN" -m llava.eval.model_vqa_loader \
      --model-path "$CHECKPOINT" \
      --model-base "$MODEL_BASE" \
      --question-file "$question_file" \
      --image-folder "$image_folder" \
      --answers-file "$out_dir/${CHUNKS}_${IDX}.jsonl" \
      --num-chunks "$CHUNKS" \
      --chunk-idx "$IDX" \
      --temperature 0 \
      --conv-mode vicuna_v1 > "$out_dir/chunk_${IDX}.log" 2>&1 &
  done
  wait
  cat "$out_dir"/"${CHUNKS}"_*.jsonl > "$out_dir/answers.jsonl"
}

if has_task iconqa; then
  run_vqa iconqa "$OFFICIAL_JSON_ROOT/IconQA_txt/iconqa_txt-test.jsonl" "$ROOT/data/iconqa/iconqa_data"
  "$PYTHON_BIN" -m llava.eval.eval_iconqa \
    --annotation-file "$OFFICIAL_JSON_ROOT/IconQA_txt/iconqa_txt-test.jsonl" \
    --result-file "$OUTPUT_ROOT/$RUN_NAME/iconqa/$CKPT_NAME/answers.jsonl" \
    --output-dir "$OUTPUT_ROOT/$RUN_NAME/iconqa/$CKPT_NAME" \
    --summary-output-dir "$SUMMARY"
fi

if has_task okvqa; then
  run_vqa okvqa "$OFFICIAL_JSON_ROOT/OKVQA/okvqa_val.jsonl" "$ROOT/data/okvqa/val2014"
  "$PYTHON_BIN" -m llava.eval.eval_okvqa \
    --annotation-file "$OFFICIAL_JSON_ROOT/OKVQA/mscoco_val2014_annotations.json" \
    --question-file "$OFFICIAL_JSON_ROOT/OKVQA/OpenEnded_mscoco_val2014_questions.json" \
    --result-file "$OUTPUT_ROOT/$RUN_NAME/okvqa/$CKPT_NAME/answers.jsonl" \
    --output-dir "$OUTPUT_ROOT/$RUN_NAME/okvqa/$CKPT_NAME" \
    --summary-output-dir "$SUMMARY"
fi

if has_task ocrvqa; then
  run_vqa ocrvqa "$OFFICIAL_JSON_ROOT/OCRVQA/sampled_ocrvqa_test.jsonl" "$ROOT/data/ocrvqa/images"
  "$PYTHON_BIN" -m llava.eval.eval_ocrvqa \
    --annotation-file "$OFFICIAL_JSON_ROOT/OCRVQA/sampled_ocrvqa_test.jsonl" \
    --result-file "$OUTPUT_ROOT/$RUN_NAME/ocrvqa/$CKPT_NAME/answers.jsonl" \
    --output-dir "$OUTPUT_ROOT/$RUN_NAME/ocrvqa/$CKPT_NAME" \
    --summary-output-dir "$SUMMARY"
fi

if has_task gqa; then
  run_vqa gqa "$OFFICIAL_JSON_ROOT/GQA/llava_gqa_testdev_balanced.jsonl" "$ROOT/data/gqa/data/images"
  "$PYTHON_BIN" -m llava.eval.eval_gqa_simple \
    --annotation-file "$ROOT/data/gqa/data/testdev_balanced_questions.json" \
    --result-file "$OUTPUT_ROOT/$RUN_NAME/gqa/$CKPT_NAME/answers.jsonl" \
    --output-dir "$OUTPUT_ROOT/$RUN_NAME/gqa/$CKPT_NAME" \
    --summary-output-dir "$SUMMARY"
fi

if has_task textvqa; then
  run_vqa textvqa "$OFFICIAL_JSON_ROOT/TextVQA/llava_textvqa_val_v051_ocr.jsonl" "$ROOT/data/textvqa/images/train_images"
  "$PYTHON_BIN" -m llava.eval.eval_textvqa \
    --annotation-file "$OFFICIAL_JSON_ROOT/TextVQA/TextVQA_0.5.1_val.json" \
    --result-file "$OUTPUT_ROOT/$RUN_NAME/textvqa/$CKPT_NAME/answers.jsonl" \
    --output-dir "$OUTPUT_ROOT/$RUN_NAME/textvqa/$CKPT_NAME" \
    --summary-output-dir "$SUMMARY"
fi

"$PYTHON_BIN" - "$RUN_NAME" "$CHECKPOINT" "$TASKS" "$SUMMARY" "$METRICS" "$LOG_FILE" <<'PY'
import json, re, subprocess, sys
run_name, ckpt, tasks, summary, metrics_path, log_file = sys.argv[1:7]
text = open(summary).read()
patterns = {
    "IconQA": r"Accuracy on IconQA:\s*([0-9.]+)%",
    "OKVQA": r"Accuracy on OKVQA:\s*([0-9.]+)%",
    "OCRVQA": r"Accuracy on OCRVQA:\s*([0-9.]+)%",
    "GQA": r"Accuracy on GQA:\s*([0-9.]+)%",
    "TextVQA": r"Accuracy on TextVQA:\s*([0-9.]+)%",
}
vals = {}
for key, pat in patterns.items():
    m = re.search(pat, text)
    vals[key] = float(m.group(1)) if m else None
target = 71.05375
req = None
if all(vals.get(k) is not None for k in ["IconQA", "OKVQA", "OCRVQA", "TextVQA"]):
    req = 8 * target - 4 * vals["IconQA"] - vals["OKVQA"] - vals["OCRVQA"] - vals["TextVQA"]
source = None
avg = None
if all(vals.get(k) is not None for k in ["OKVQA", "OCRVQA", "GQA", "TextVQA"]):
    source = sum(vals[k] for k in ["OKVQA", "OCRVQA", "GQA", "TextVQA"]) / 4
if source is not None and vals.get("IconQA") is not None:
    avg = (source + vals["IconQA"]) / 2
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = ""
promote = bool(
    req is not None and req <= 57.0
    and vals["IconQA"] >= 86.20
    and vals["OKVQA"] >= 52.50
    and vals["OCRVQA"] >= 55.50
    and vals["TextVQA"] >= 52.00
)
out = {
    "run_name": run_name,
    "checkpoint": ckpt,
    "tasks_completed": [t for t in tasks.split(",") if t],
    "metrics": {**vals, "SourceAvg": source, "Avg": avg},
    "required_gqa_for_target": req,
    "target_avg": target,
    "promote_to_full_eval": promote,
    "git_commit": commit,
    "log_file": log_file,
}
with open(metrics_path, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PY
