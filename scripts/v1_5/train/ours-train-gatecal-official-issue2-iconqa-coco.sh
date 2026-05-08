#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/data/guoboyang/LoRa-Projects/LoRASculpt-repro}
CODE_DIR=${CODE_DIR:-$ROOT/LoRASculpt}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$ROOT/checkpoints}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

export DEVICE=${DEVICE:-localhost:0,1,2,3}
export MASTER_PORT=${MASTER_PORT:-29670}
export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
export DEEPSPEED_ZEROFILE=${DEEPSPEED_ZEROFILE:-"$CODE_DIR/scripts/zero2.json"}
export DEEPSPEED_BIN=${DEEPSPEED_BIN:-/data/guoboyang/miniconda3/envs/lorasculpt/bin/deepspeed}
export USE_DEEPSPEED=${USE_DEEPSPEED:-0}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LEARNING_RATE=${LEARNING_RATE:-5e-2}
export MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-False}

export GATECAL_INIT=${GATECAL_INIT:-0.95}
export GATECAL_ANCHOR=${GATECAL_ANCHOR:-1.0}
export GATECAL_REG_LAMBDA=${GATECAL_REG_LAMBDA:-1e-3}

export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-$ROOT/models/llava-v1.5-7b-ft}
export VISION_TOWER=${VISION_TOWER:-$ROOT/models/clip-vit-large-patch14-336}
export BASELINE_LORA=${BASELINE_LORA:-$CHECKPOINT_ROOT/llava-v1.5-7b-iconqa_txt_official_issue2-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1}
export DATA_PATH_OVERRIDE=${DATA_PATH_OVERRIDE:-$ROOT/data/samix/iconqa_official_issue2_coco1500_seed42.json}
export IMAGE_FOLDER_OVERRIDE=${IMAGE_FOLDER_OVERRIDE:-$ROOT/data}
export RUN_NAME=${RUN_NAME:-llava-v1.5-7b-lorasculpt-gatecal-coco1500-r32}

extra_args=()
deepspeed_args=()
if [[ "$USE_DEEPSPEED" == "1" ]]; then
    deepspeed_args+=(--deepspeed "$DEEPSPEED_ZEROFILE")
fi
OUTPUT_DIR_SUFFIX=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    : ${MAX_STEPS:=20}
    OUTPUT_DIR_SUFFIX="-smoke"
    extra_args+=(--max_steps "$MAX_STEPS" --save_steps "$MAX_STEPS")
elif [[ -n "${MAX_STEPS:-}" ]]; then
    extra_args+=(--max_steps "$MAX_STEPS")
fi

OUTPUT_DIR=${OUTPUT_DIR:-$CHECKPOINT_ROOT/${RUN_NAME}${OUTPUT_DIR_SUFFIX}}

echo "[GateCal] data_path=$DATA_PATH_OVERRIDE"
echo "[GateCal] image_folder=$IMAGE_FOLDER_OVERRIDE"
echo "[GateCal] output_dir=$OUTPUT_DIR"
echo "[GateCal] baseline_lora=$BASELINE_LORA"
echo "[GateCal] gate_init=$GATECAL_INIT anchor=$GATECAL_ANCHOR reg=$GATECAL_REG_LAMBDA lr=$LEARNING_RATE use_deepspeed=$USE_DEEPSPEED extra_args=${extra_args[*]:-}"

cd "$CODE_DIR"
"$DEEPSPEED_BIN" --include "$DEVICE" --master_port "$MASTER_PORT" llava/train/train_mem.py \
    --lora_enable True --lora_r "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --lora_start_path "$BASELINE_LORA" --mm_projector_lr 2e-5 \
    "${deepspeed_args[@]}" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --version v1 \
    --data_path "$DATA_PATH_OVERRIDE" \
    --image_folder "$IMAGE_FOLDER_OVERRIDE" \
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
    --save_total_limit 5 \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps "${LOGGING_STEPS:-1}" \
    --tf32 True \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --lazy_preprocess True \
    --report_to none \
    --trainer_name LoRASculptGateCal \
    "${extra_args[@]}"
