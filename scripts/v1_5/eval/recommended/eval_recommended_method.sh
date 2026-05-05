#!/bin/bash
set -euo pipefail

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

: ${MODEL_PATH:="${1:-}"}
: ${TARGET_DATASET:="${2:-}"}   # iconqa_txt | coco
: ${METHOD_NAME:="${3:-recommended}"}
: ${MODEL_BASE:=/data/guoboyang/LoRa-Projects/LoRASculpt-repro/models/llava-v1.5-7b-ft}
: ${PYTHON_BIN:=/data/guoboyang/miniconda3/envs/lorasculpt/bin/python}
: ${CUDA_VISIBLE_DEVICES:=0,1,2,3}
: ${RESULT_ROOT:=/data/guoboyang/LoRa-Projects/LoRASculpt-repro/repro_results/recommended_methods/${METHOD_NAME}-${TARGET_DATASET}}

if [ -z "$MODEL_PATH" ] || [ -z "$TARGET_DATASET" ]; then
    echo "Usage: MODEL_PATH=/path TARGET_DATASET=iconqa_txt|coco METHOD_NAME=name bash $0" >&2
    exit 1
fi

IFS=, read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}
CKPT="llava-v1.5-7b"
SUMMARY_OUTPUT_DIR="$RESULT_ROOT/summary.txt"
mkdir -p "$RESULT_ROOT"
> "$SUMMARY_OUTPUT_DIR"
{
  echo "Method: $METHOD_NAME"
  echo "Model: $MODEL_PATH"
  echo "Model base: $MODEL_BASE"
  echo "Target dataset: $TARGET_DATASET"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "Started: $(date -Is)"
  echo ""
} >> "$SUMMARY_OUTPUT_DIR"

model_base_args=()
case "${MODEL_BASE,,}" in
  ""|none|null)
    ;;
  *)
    model_base_args+=(--model-base "$MODEL_BASE")
    ;;
esac

run_vqa() {
    local split="$1"
    local question_file="$2"
    local image_folder="$3"
    local out_dir="$RESULT_ROOT/$split/$CKPT"
    mkdir -p "$out_dir"
    rm -f "$out_dir"/"${CHUNKS}"_*.jsonl "$out_dir/answers.jsonl"
    echo "[$split] question_file=$question_file"
    echo "[$split] image_folder=$image_folder"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} "$PYTHON_BIN" -m llava.eval.model_vqa_loader \
            --model-path "$MODEL_PATH" \
            "${model_base_args[@]}" \
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

run_caption() {
    local split="$1"
    local question_file="$2"
    local image_folder="$3"
    local out_dir="$RESULT_ROOT/$split/$CKPT"
    mkdir -p "$out_dir"
    rm -f "$out_dir"/"${CHUNKS}"_*.jsonl "$out_dir/answers.jsonl"
    echo "[$split] question_file=$question_file"
    echo "[$split] image_folder=$image_folder"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} "$PYTHON_BIN" -m llava.eval.model_caption_loader \
            --model-path "$MODEL_PATH" \
            "${model_base_args[@]}" \
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

if [ "$TARGET_DATASET" = "iconqa_txt" ]; then
    run_vqa iconqa \
      /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/iconqa/fixed_iconqa_txt-test.jsonl \
      /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/iconqa
    "$PYTHON_BIN" -m llava.eval.eval_iconqa \
      --annotation-file /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/iconqa/fixed_iconqa_txt-test.jsonl \
      --result-file "$RESULT_ROOT/iconqa/$CKPT/answers.jsonl" \
      --output-dir "$RESULT_ROOT/iconqa/$CKPT" \
      --summary-output-dir "$SUMMARY_OUTPUT_DIR"
elif [ "$TARGET_DATASET" = "coco" ]; then
    run_caption coco \
      /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/coco/llava_caption_mscoco_test.jsonl \
      /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/coco/val2014
    "$PYTHON_BIN" -m llava.eval.eval_caption \
      --annotation-file /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/coco/captions_test5k.json \
      --result-file "$RESULT_ROOT/coco/$CKPT/answers.jsonl" \
      --output-dir "$RESULT_ROOT/coco/$CKPT" \
      --summary-output-dir "$SUMMARY_OUTPUT_DIR"
else
    echo "Unsupported TARGET_DATASET: $TARGET_DATASET" >&2
    exit 1
fi

run_vqa okvqa \
  /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/okvqa/okvqa_val.jsonl \
  /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/okvqa/val2014
"$PYTHON_BIN" -m llava.eval.eval_okvqa \
  --annotation-file /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/okvqa/mscoco_val2014_annotations.json \
  --question-file /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/okvqa/OpenEnded_mscoco_val2014_questions.json \
  --result-file "$RESULT_ROOT/okvqa/$CKPT/answers.jsonl" \
  --output-dir "$RESULT_ROOT/okvqa/$CKPT" \
  --summary-output-dir "$SUMMARY_OUTPUT_DIR"

run_vqa ocrvqa \
  /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/ocrvqa/sampled_ocrvqa_test.jsonl \
  /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/ocrvqa/sampled_images
"$PYTHON_BIN" -m llava.eval.eval_ocrvqa \
  --annotation-file /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/ocrvqa/sampled_ocrvqa_test.jsonl \
  --result-file "$RESULT_ROOT/ocrvqa/$CKPT/answers.jsonl" \
  --output-dir "$RESULT_ROOT/ocrvqa/$CKPT" \
  --summary-output-dir "$SUMMARY_OUTPUT_DIR"

run_vqa gqa \
  /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/gqa/llava_gqa_testdev_balanced.jsonl \
  /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/gqa/data/images
"$PYTHON_BIN" scripts/convert_gqa_for_eval.py \
  --src "$RESULT_ROOT/gqa/$CKPT/answers.jsonl" \
  --dst /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/gqa/data/testdev_balanced_predictions.json
"$PYTHON_BIN" -m llava.eval.eval_gqa_simple \
  --annotation-file /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/gqa/data/testdev_balanced_questions.json \
  --result-file "$RESULT_ROOT/gqa/$CKPT/answers.jsonl" \
  --output-dir "$RESULT_ROOT/gqa/$CKPT" \
  --summary-output-dir "$SUMMARY_OUTPUT_DIR"

run_vqa textvqa \
  /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/textvqa/llava_textvqa_val_v051_ocr.jsonl \
  /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/textvqa/images/train_images
"$PYTHON_BIN" -m llava.eval.eval_textvqa \
  --annotation-file /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/textvqa/TextVQA_0.5.1_val.json \
  --result-file "$RESULT_ROOT/textvqa/$CKPT/answers.jsonl" \
  --output-dir "$RESULT_ROOT/textvqa/$CKPT" \
  --summary-output-dir "$SUMMARY_OUTPUT_DIR"

"$PYTHON_BIN" scripts/v1_5/recommended_methods/summarize_baseline_table.py \
  --summary-file "$SUMMARY_OUTPUT_DIR" \
  --target "$TARGET_DATASET" \
  --method "$METHOD_NAME" \
  --output-json "$RESULT_ROOT/baseline_row.json" \
  --output-md "$RESULT_ROOT/baseline_row.md"

cat "$SUMMARY_OUTPUT_DIR"
echo "Baseline row: $RESULT_ROOT/baseline_row.md"
