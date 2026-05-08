#!/bin/bash
set -euo pipefail

LORA_CHECKPOINT="${LORA_CHECKPOINT:-}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"
RUN_NAME="${RUN_NAME:-}"
TASKS="${TASKS:-iconqa,okvqa,ocrvqa,gqa,textvqa}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
LOG_FILE="${LOG_FILE:-}"
MAX_QUESTIONS_PER_TASK="${MAX_QUESTIONS_PER_TASK:-0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --lora-checkpoint) LORA_CHECKPOINT="$2"; shift 2 ;;
    --base-checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --max-questions-per-task) MAX_QUESTIONS_PER_TASK="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

: "${ROOT:=/data/guoboyang/LoRa-Projects/LoRASculpt-repro}"
: "${MODEL_BASE:=$ROOT/models/llava-v1.5-7b-ft}"
: "${BASE_CHECKPOINT:=$ROOT/models/llava-v1.5-7b-ft}"
: "${PYTHON_BIN:=/data/guoboyang/miniconda3/envs/lorasculpt/bin/python}"
: "${OFFICIAL_JSON_ROOT:=$ROOT/downloads/official_issue2/extracted/LoRASculpt_JSON_files}"
: "${OUTPUT_ROOT:=$ROOT/repro_results/prompt_form_gate}"

if [ -z "$LORA_CHECKPOINT" ] || [ -z "$RUN_NAME" ]; then
  echo "Usage: $0 --lora-checkpoint CKPT --run-name RUN [--base-checkpoint CKPT] [--tasks iconqa,okvqa,ocrvqa,gqa,textvqa]" >&2
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
  echo "[prompt_form_gate_eval] log_file=$LOG_FILE"
fi

echo "LoRA model: $LORA_CHECKPOINT" > "$SUMMARY"
echo "Base model: $BASE_CHECKPOINT" >> "$SUMMARY"
echo "Official JSON root: $OFFICIAL_JSON_ROOT" >> "$SUMMARY"
echo "LoRA model base: $MODEL_BASE" >> "$SUMMARY"
echo "Gate: prompt-form MCQA -> LoRA, open-ended VQA -> base" >> "$SUMMARY"
echo "" >> "$SUMMARY"

has_task() {
  case ",$TASKS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

model_base_args_for() {
  local model_base="$1"
  case "${model_base,,}" in
    ""|none|null) ;;
    *) printf '%s\n' --model-base "$model_base" ;;
  esac
}

run_vqa_with_model() {
  local split="$1"
  local gate_name="$2"
  local question_file="$3"
  local image_folder="$4"
  local model_path="$5"
  local model_base="$6"
  local out_dir="$OUTPUT_ROOT/$RUN_NAME/$split/$CKPT_NAME"
  local answers_file="$out_dir/${gate_name}_answers.jsonl"
  local count

  count=$(wc -l < "$question_file")
  mkdir -p "$out_dir"
  rm -f "$out_dir"/"${gate_name}_${CHUNKS}"_*.jsonl "$answers_file"
  if [ "$count" -eq 0 ]; then
    : > "$answers_file"
    return 0
  fi

  echo "[$split][$gate_name] model=$model_path"
  echo "[$split][$gate_name] question_file=$question_file"
  echo "[$split][$gate_name] image_folder=$image_folder"

  local model_base_args=()
  while IFS= read -r arg; do
    [ -n "$arg" ] && model_base_args+=("$arg")
  done < <(model_base_args_for "$model_base")

  for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} "$PYTHON_BIN" -m llava.eval.model_vqa_loader \
      --model-path "$model_path" \
      "${model_base_args[@]}" \
      --question-file "$question_file" \
      --image-folder "$image_folder" \
      --answers-file "$out_dir/${gate_name}_${CHUNKS}_${IDX}.jsonl" \
      --num-chunks "$CHUNKS" \
      --chunk-idx "$IDX" \
      --temperature 0 \
      --conv-mode vicuna_v1 > "$out_dir/${gate_name}_chunk_${IDX}.log" 2>&1 &
  done
  wait
  cat "$out_dir"/"${gate_name}_${CHUNKS}"_*.jsonl > "$answers_file"
}

run_prompt_form_gate_vqa() {
  local split="$1"
  local question_file="$2"
  local image_folder="$3"
  local out_dir="$OUTPUT_ROOT/$RUN_NAME/$split/$CKPT_NAME"
  local split_dir="$out_dir/gate_split"

  mkdir -p "$out_dir" "$split_dir"
  "$PYTHON_BIN" scripts/v1_5/tools/prompt_form_gate_split_questions.py \
    --question-file "$question_file" \
    --output-dir "$split_dir" \
    --limit "$MAX_QUESTIONS_PER_TASK"

  run_vqa_with_model "$split" gate_on "$split_dir/gate_on.jsonl" "$image_folder" "$LORA_CHECKPOINT" "$MODEL_BASE"
  run_vqa_with_model "$split" gate_off "$split_dir/gate_off.jsonl" "$image_folder" "$BASE_CHECKPOINT" none

  local merge_question_file="$question_file"
  if [ "$MAX_QUESTIONS_PER_TASK" != "0" ]; then
    merge_question_file="$split_dir/limited_questions.jsonl"
    head -n "$MAX_QUESTIONS_PER_TASK" "$question_file" > "$merge_question_file"
  fi

  "$PYTHON_BIN" scripts/v1_5/tools/prompt_form_gate_merge_answers.py \
    --question-file "$merge_question_file" \
    --lora-answers "$out_dir/gate_on_answers.jsonl" \
    --base-answers "$out_dir/gate_off_answers.jsonl" \
    --output-answers "$out_dir/answers.jsonl" \
    --stats-output "$out_dir/gate_stats.json"
}

if has_task iconqa; then
  run_prompt_form_gate_vqa iconqa "$OFFICIAL_JSON_ROOT/IconQA_txt/iconqa_txt-test.jsonl" "$ROOT/data/iconqa/iconqa_data"
  "$PYTHON_BIN" -m llava.eval.eval_iconqa \
    --annotation-file "$OFFICIAL_JSON_ROOT/IconQA_txt/iconqa_txt-test.jsonl" \
    --result-file "$OUTPUT_ROOT/$RUN_NAME/iconqa/$CKPT_NAME/answers.jsonl" \
    --output-dir "$OUTPUT_ROOT/$RUN_NAME/iconqa/$CKPT_NAME" \
    --summary-output-dir "$SUMMARY"
fi

if has_task okvqa; then
  run_prompt_form_gate_vqa okvqa "$OFFICIAL_JSON_ROOT/OKVQA/okvqa_val.jsonl" "$ROOT/data/okvqa/val2014"
  if [ "$MAX_QUESTIONS_PER_TASK" = "0" ]; then
    "$PYTHON_BIN" -m llava.eval.eval_okvqa \
      --annotation-file "$OFFICIAL_JSON_ROOT/OKVQA/mscoco_val2014_annotations.json" \
      --question-file "$OFFICIAL_JSON_ROOT/OKVQA/OpenEnded_mscoco_val2014_questions.json" \
      --result-file "$OUTPUT_ROOT/$RUN_NAME/okvqa/$CKPT_NAME/answers.jsonl" \
      --output-dir "$OUTPUT_ROOT/$RUN_NAME/okvqa/$CKPT_NAME" \
      --summary-output-dir "$SUMMARY"
  else
    echo "" >> "$SUMMARY"
    echo "Subset smoke on OKVQA: ${MAX_QUESTIONS_PER_TASK} requested samples; official OKVQA eval skipped because the VQA API requires the complete question set." >> "$SUMMARY"
  fi
fi

if has_task ocrvqa; then
  run_prompt_form_gate_vqa ocrvqa "$OFFICIAL_JSON_ROOT/OCRVQA/sampled_ocrvqa_test.jsonl" "$ROOT/data/ocrvqa/images"
  "$PYTHON_BIN" -m llava.eval.eval_ocrvqa \
    --annotation-file "$OFFICIAL_JSON_ROOT/OCRVQA/sampled_ocrvqa_test.jsonl" \
    --result-file "$OUTPUT_ROOT/$RUN_NAME/ocrvqa/$CKPT_NAME/answers.jsonl" \
    --output-dir "$OUTPUT_ROOT/$RUN_NAME/ocrvqa/$CKPT_NAME" \
    --summary-output-dir "$SUMMARY"
fi

if has_task gqa; then
  run_prompt_form_gate_vqa gqa "$OFFICIAL_JSON_ROOT/GQA/llava_gqa_testdev_balanced.jsonl" "$ROOT/data/gqa/data/images"
  "$PYTHON_BIN" -m llava.eval.eval_gqa_simple \
    --annotation-file "$ROOT/data/gqa/data/testdev_balanced_questions.json" \
    --result-file "$OUTPUT_ROOT/$RUN_NAME/gqa/$CKPT_NAME/answers.jsonl" \
    --output-dir "$OUTPUT_ROOT/$RUN_NAME/gqa/$CKPT_NAME" \
    --summary-output-dir "$SUMMARY"
fi

if has_task textvqa; then
  run_prompt_form_gate_vqa textvqa "$OFFICIAL_JSON_ROOT/TextVQA/llava_textvqa_val_v051_ocr.jsonl" "$ROOT/data/textvqa/images/train_images"
  "$PYTHON_BIN" -m llava.eval.eval_textvqa \
    --annotation-file "$OFFICIAL_JSON_ROOT/TextVQA/TextVQA_0.5.1_val.json" \
    --result-file "$OUTPUT_ROOT/$RUN_NAME/textvqa/$CKPT_NAME/answers.jsonl" \
    --output-dir "$OUTPUT_ROOT/$RUN_NAME/textvqa/$CKPT_NAME" \
    --summary-output-dir "$SUMMARY"
fi

"$PYTHON_BIN" - "$RUN_NAME" "$LORA_CHECKPOINT" "$BASE_CHECKPOINT" "$TASKS" "$SUMMARY" "$METRICS" "$LOG_FILE" <<'PY'
import json, re, subprocess, sys

run_name, lora_ckpt, base_ckpt, tasks, summary, metrics_path, log_file = sys.argv[1:8]
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
out = {
    "run_name": run_name,
    "method": "Prompt-Form Adaptive Residual Gate runtime eval",
    "lora_checkpoint": lora_ckpt,
    "base_checkpoint": base_ckpt,
    "tasks": tasks,
    "metrics": {**vals, "SourceAvg": source, "Avg": avg},
    "delta_vs_exact_baseline": None if avg is None else avg - 70.05375,
    "target_avg": 71.05375,
    "promote": bool(avg is not None and avg >= 71.05375),
    "git_commit": commit,
    "log_file": log_file,
}
with open(metrics_path, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
PY
