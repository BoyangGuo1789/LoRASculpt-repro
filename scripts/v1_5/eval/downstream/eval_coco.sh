#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


MODEL_PATH=""
MODEL_BASE="/data/guoboyang/LoRa-Projects/LoRASculpt-repro/models/llava-v1.5-7b-ft"
CKPT="llava-v1.5-7b"
SPLIT="coco"
RESULT_DIR=""


if [ ! -n "$1" ] ;then
    MODEL_PATH=$MODEL_PATH
else
    MODEL_PATH=$1
fi

if [ ! -n "$2" ] ;then
    RESULT_DIR=$RESULT_DIR
else
    RESULT_DIR=$2
fi

if [ ! -n "$3" ] ;then
    SUMMARY_OUTPUT_DIR="None"
else
    SUMMARY_OUTPUT_DIR=$3
fi

mkdir -p $RESULT_DIR





for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_caption_loader \
        --model-path $MODEL_PATH \
        --model-base $MODEL_BASE \
        --question-file /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/coco/llava_caption_mscoco_test.jsonl \
        --image-folder /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/coco/val2014 \
        --answers-file $RESULT_DIR/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done




wait



output_file=$RESULT_DIR/$SPLIT/$CKPT/llava-v1.5-7b-coco.jsonl



# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

CAPTION_EVAL_PYTHON=${CAPTION_EVAL_PYTHON:-python}
if [ "$CAPTION_EVAL_PYTHON" = "python" ]; then
    python -m llava.eval.eval_caption \
        --annotation-file /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/coco/captions_test5k.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR/$SPLIT/$CKPT \
        --summary-output-dir $SUMMARY_OUTPUT_DIR
else
    $CAPTION_EVAL_PYTHON llava/eval/eval_caption.py \
        --annotation-file /data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/coco/captions_test5k.json \
        --result-file $output_file \
        --output-dir $RESULT_DIR/$SPLIT/$CKPT \
        --summary-output-dir $SUMMARY_OUTPUT_DIR
fi
