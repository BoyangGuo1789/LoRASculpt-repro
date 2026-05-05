#!/bin/bash
set -euo pipefail

: ${METHOD:=""}          # spider | adadare_gamma
: ${DATASET_NAME:=""}    # iconqa_txt | coco
: ${PYTHON_BIN:=/data/guoboyang/miniconda3/envs/lorasculpt/bin/python}
: ${BASE_MODEL:=/data/guoboyang/LoRa-Projects/LoRASculpt-repro/models/llava-v1.5-7b-ft}
: ${CHECKPOINT_ROOT:=/data/guoboyang/LoRa-Projects/LoRASculpt-repro/checkpoints}

if [ -z "$METHOD" ] || [ -z "$DATASET_NAME" ]; then
    echo "Usage: METHOD=spider|adadare_gamma DATASET_NAME=iconqa_txt|coco bash $0" >&2
    exit 1
fi

case "$METHOD" in
  spider)
    : ${OUTPUT_DIR:=$CHECKPOINT_ROOT/llava-v1.5-7b-${DATASET_NAME}-SPIDER-recommended-e${NUM_TRAIN_EPOCHS:-5}-last${TUNE_DECODER_LAYER:-2}}
    export OUTPUT_DIR DATASET_NAME
    bash scripts/v1_5/train/trainconfig_spider.sh
    echo "$OUTPUT_DIR"
    ;;
  adadare_gamma)
    : ${ADAPTER_OUTPUT_DIR:=$CHECKPOINT_ROOT/llava-v1.5-7b-${DATASET_NAME}-AdaDARE-gamma-lora-r${LORA_RANK:-128}-a${LORA_ALPHA:-256}-e${NUM_TRAIN_EPOCHS:-5}}
    : ${FUSED_OUTPUT_DIR:=${ADAPTER_OUTPUT_DIR}-fused-gamma${ADADARE_GAMMA:-0.7}-sparsity${ADADARE_SPARSITY:-0.9}}
    export OUTPUT_DIR="$ADAPTER_OUTPUT_DIR"
    export DATASET_NAME
    bash scripts/v1_5/train/trainconfig_adadare_lora.sh
    "$PYTHON_BIN" scripts/v1_5/recommended_methods/adadare_gamma_fuse.py \
      --adapter-path "$ADAPTER_OUTPUT_DIR" \
      --base-model "$BASE_MODEL" \
      --output-dir "$FUSED_OUTPUT_DIR" \
      --gamma "${ADADARE_GAMMA:-0.7}" \
      --sparsity "${ADADARE_SPARSITY:-0.9}" \
      --seed "${ADADARE_SEED:-42}"
    echo "$FUSED_OUTPUT_DIR"
    ;;
  *)
    echo "Unsupported METHOD: $METHOD" >&2
    exit 1
    ;;
esac
