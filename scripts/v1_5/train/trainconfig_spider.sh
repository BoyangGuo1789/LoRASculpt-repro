#!/bin/bash
set -euo pipefail

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

: ${DEVICE:=localhost:0,1,2,3,4,5,6,7}
: ${MASTER_PORT:=29610}
: ${PER_DEVICE_TRAIN_BATCH_SIZE:=2}
: ${GRADIENT_ACCUMULATION_STEPS:=1}
: ${NUM_TRAIN_EPOCHS:=5}
: ${LEARNING_RATE:=2e-4}
: ${MM_PROJECTOR_LR:=2e-5}
: ${TUNE_DECODER_LAYER:=2}
: ${TRAINER_NAME:=SPIDER}
: ${DATASET_NAME:=""}
: ${OUTPUT_DIR:=""}
: ${MODEL_NAME_OR_PATH:=/data/guoboyang/LoRa-Projects/LoRASculpt-repro/models/llava-v1.5-7b-ft}
: ${VISION_TOWER:=/data/guoboyang/LoRa-Projects/LoRASculpt-repro/models/clip-vit-large-patch14-336}
: ${DEEPSPEED_ZEROFILE:=./scripts/zero2.json}
: ${DEEPSPEED_BIN:=/data/guoboyang/miniconda3/envs/lorasculpt/bin/deepspeed}

if [ "$DATASET_NAME" == "iconqa_txt" ]; then
    data_path="/data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/iconqa/iconqa_txt-train.json"
    image_folder="/data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/iconqa"
elif [ "$DATASET_NAME" == "coco" ]; then
    data_path="/data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/coco/coco-train.json"
    image_folder="/data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/coco"
else
    echo "Unsupported DATASET_NAME: $DATASET_NAME" >&2
    exit 1
fi

if [ -n "${DATA_PATH_OVERRIDE:-}" ]; then
    data_path="$DATA_PATH_OVERRIDE"
fi
if [ -n "${IMAGE_FOLDER_OVERRIDE:-}" ]; then
    image_folder="$IMAGE_FOLDER_OVERRIDE"
fi
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/data/guoboyang/LoRa-Projects/LoRASculpt-repro/checkpoints/llava-v1.5-7b-${DATASET_NAME}-SPIDER-recommended-e${NUM_TRAIN_EPOCHS}-last${TUNE_DECODER_LAYER}"
fi

extra_args=()
if [ -n "${MAX_STEPS:-}" ]; then
    extra_args+=(--max_steps "$MAX_STEPS")
fi

echo "[SPIDER] data_path=$data_path"
echo "[SPIDER] image_folder=$image_folder"
echo "[SPIDER] output_dir=$OUTPUT_DIR"
echo "[SPIDER] trainer=$TRAINER_NAME tune_mm_mlp_adapter=True tune_decoder_layer=$TUNE_DECODER_LAYER lr=$LEARNING_RATE mm_projector_lr=$MM_PROJECTOR_LR epochs=$NUM_TRAIN_EPOCHS"

"$DEEPSPEED_BIN" --include "$DEVICE" --master_port "$MASTER_PORT" llava/train/train_mem.py \
    --deepspeed "$DEEPSPEED_ZEROFILE" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --version v1 \
    --data_path "$data_path" \
    --image_folder "$image_folder" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 15 \
    --learning_rate "$LEARNING_RATE" \
    --mm_projector_lr "$MM_PROJECTOR_LR" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --tune_mm_mlp_adapter True \
    --tune_decoder_layer "$TUNE_DECODER_LAYER" \
    --trainer_name "$TRAINER_NAME" \
    "${extra_args[@]}"
