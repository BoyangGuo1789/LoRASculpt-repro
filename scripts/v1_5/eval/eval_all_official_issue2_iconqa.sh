#!/bin/bash

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

MODEL_PATH="${1:-/data/guoboyang/LoRa-Projects/LoRASculpt-repro/checkpoints/llava-v1.5-7b-iconqa_txt_official_issue2-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1}"
MODEL_BASE="${MODEL_BASE:-/data/guoboyang/LoRa-Projects/LoRASculpt-repro/models/llava-v1.5-7b-ft}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT="llava-v1.5-7b"
RESULT_ROOT="${RESULT_ROOT:-/data/guoboyang/LoRa-Projects/LoRASculpt-repro/repro_results/20260505_lorasculpt_iconqa_official_issue2_rank32}"
SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-$RESULT_ROOT/summary.txt}"
OFFICIAL_JSON_ROOT="${OFFICIAL_JSON_ROOT:-/data/guoboyang/LoRa-Projects/LoRASculpt-repro/downloads/official_issue2/extracted/LoRASculpt_JSON_files}"

mkdir -p "$RESULT_ROOT"
> "$SUMMARY_OUTPUT_DIR"
echo "Model: $MODEL_PATH" >> "$SUMMARY_OUTPUT_DIR"
echo "Official JSON root: $OFFICIAL_JSON_ROOT" >> "$SUMMARY_OUTPUT_DIR"

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

run_vqa iconqa \
    "$OFFICIAL_JSON_ROOT/IconQA_txt/iconqa_txt-test.jsonl" \
    /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/iconqa/iconqa_data
"$PYTHON_BIN" -m llava.eval.eval_iconqa \
    --annotation-file "$OFFICIAL_JSON_ROOT/IconQA_txt/iconqa_txt-test.jsonl" \
    --result-file "$RESULT_ROOT/iconqa/$CKPT/answers.jsonl" \
    --output-dir "$RESULT_ROOT/iconqa/$CKPT" \
    --summary-output-dir "$SUMMARY_OUTPUT_DIR"

run_vqa okvqa \
    "$OFFICIAL_JSON_ROOT/OKVQA/okvqa_val.jsonl" \
    /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/okvqa/val2014
"$PYTHON_BIN" -m llava.eval.eval_okvqa \
    --annotation-file "$OFFICIAL_JSON_ROOT/OKVQA/mscoco_val2014_annotations.json" \
    --question-file "$OFFICIAL_JSON_ROOT/OKVQA/OpenEnded_mscoco_val2014_questions.json" \
    --result-file "$RESULT_ROOT/okvqa/$CKPT/answers.jsonl" \
    --output-dir "$RESULT_ROOT/okvqa/$CKPT" \
    --summary-output-dir "$SUMMARY_OUTPUT_DIR"

run_vqa ocrvqa \
    "$OFFICIAL_JSON_ROOT/OCRVQA/sampled_ocrvqa_test.jsonl" \
    /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/ocrvqa/images
"$PYTHON_BIN" -m llava.eval.eval_ocrvqa \
    --annotation-file "$OFFICIAL_JSON_ROOT/OCRVQA/sampled_ocrvqa_test.jsonl" \
    --result-file "$RESULT_ROOT/ocrvqa/$CKPT/answers.jsonl" \
    --output-dir "$RESULT_ROOT/ocrvqa/$CKPT" \
    --summary-output-dir "$SUMMARY_OUTPUT_DIR"

run_vqa gqa \
    "$OFFICIAL_JSON_ROOT/GQA/llava_gqa_testdev_balanced.jsonl" \
    /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/gqa/data/images
"$PYTHON_BIN" -m llava.eval.eval_gqa_simple \
    --annotation-file /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/gqa/data/testdev_balanced_questions.json \
    --result-file "$RESULT_ROOT/gqa/$CKPT/answers.jsonl" \
    --output-dir "$RESULT_ROOT/gqa/$CKPT" \
    --summary-output-dir "$SUMMARY_OUTPUT_DIR"

run_vqa textvqa \
    "$OFFICIAL_JSON_ROOT/TextVQA/llava_textvqa_val_v051_ocr.jsonl" \
    /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/textvqa/images/train_images
"$PYTHON_BIN" -m llava.eval.eval_textvqa \
    --annotation-file "$OFFICIAL_JSON_ROOT/TextVQA/TextVQA_0.5.1_val.json" \
    --result-file "$RESULT_ROOT/textvqa/$CKPT/answers.jsonl" \
    --output-dir "$RESULT_ROOT/textvqa/$CKPT" \
    --summary-output-dir "$SUMMARY_OUTPUT_DIR"

cat "$SUMMARY_OUTPUT_DIR"
