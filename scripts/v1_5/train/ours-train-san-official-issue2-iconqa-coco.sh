#!/bin/bash
set -euo pipefail

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

ROOT=${ROOT:-/data/guoboyang/LoRa-Projects/LoRASculpt-repro}
CODE_DIR=${CODE_DIR:-$ROOT/LoRASculpt}
PYTHON_BIN=${PYTHON_BIN:-/data/guoboyang/miniconda3/envs/lorasculpt/bin/python}
DEEPSPEED_BIN=${DEEPSPEED_BIN:-/data/guoboyang/miniconda3/envs/lorasculpt/bin/deepspeed}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-$ROOT/models/llava-v1.5-7b-ft}
VISION_TOWER=${VISION_TOWER:-$ROOT/models/clip-vit-large-patch14-336}

DEVICE=${DEVICE:-localhost:0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29730}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
DEEPSPEED_ZEROFILE=${DEEPSPEED_ZEROFILE:-./scripts/zero2.json}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-${EPOCHS:-3}}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
AB_PRESERVE_RATIO=${AB_PRESERVE_RATIO:-0.1}
OMEGA=${OMEGA:-1.0}
CMR_LAMBDA=${CMR_LAMBDA:-1e-3}
LEARNING_RATE=${LEARNING_RATE:-${LR:-2e-4}}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
LOGGING_STEPS=${LOGGING_STEPS:-1}
STEP_THRESHOLD=${STEP_THRESHOLD:-100}

COCO_SAMPLES=${COCO_SAMPLES:-${COCO_N:-3000}}
SAMIX_SEED=${SAMIX_SEED:-42}
SAMIX_SHUFFLE_SEED=${SAMIX_SHUFFLE_SEED:-$SAMIX_SEED}
SAMIX_DATA_DIR=${SAMIX_DATA_DIR:-$ROOT/data/samix}
OFFICIAL_JSON_ROOT=${OFFICIAL_JSON_ROOT:-$ROOT/downloads/official_issue2/extracted/LoRASculpt_JSON_files}
ICONQA_JSON=${ICONQA_JSON:-$OFFICIAL_JSON_ROOT/IconQA_txt/iconqa_txt-train.json}
COCO_JSON=${COCO_JSON:-$OFFICIAL_JSON_ROOT/COCO-Caption/coco-train.json}
SAMIX_JSON=${SAMIX_JSON:-$SAMIX_DATA_DIR/iconqa_official_issue2_coco${COCO_SAMPLES}_seed${SAMIX_SEED}.json}
SAMIX_MANIFEST=${SAMIX_MANIFEST:-$SAMIX_DATA_DIR/iconqa_official_issue2_coco${COCO_SAMPLES}_seed${SAMIX_SEED}.manifest.json}
IMAGE_FOLDER_OVERRIDE=${IMAGE_FOLDER_OVERRIDE:-$ROOT/data}
REBUILD_SAMIX=${REBUILD_SAMIX:-0}

SAN_LAMBDA=${SAN_LAMBDA:-0.05}
SAN_WARMUP_STEPS=${SAN_WARMUP_STEPS:-100}
SAN_SCOPE=${SAN_SCOPE:-qkv}
SAN_LOG_EVERY=${SAN_LOG_EVERY:-20}
SAN_EPS=${SAN_EPS:-1e-8}

TRAINER_NAME=${TRAINER_NAME:-SANLoRA}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/checkpoints}
RUN_NAME=${RUN_NAME:-llava-v1.5-7b-lorasculpt-san-coco${COCO_SAMPLES}-l${SAN_LAMBDA}-${SAN_SCOPE}-r${LORA_RANK}}
OUTPUT_DIR_SUFFIX=""
extra_args=()

if [[ "${SMOKE:-0}" == "1" ]]; then
    STEP_THRESHOLD=${STEP_THRESHOLD_OVERRIDE:-1}
    MAX_STEPS=${MAX_STEPS:-${SMOKE_MAX_STEPS:-20}}
    OUTPUT_DIR_SUFFIX="-$(date +%Y%m%d_%H%M%S)-smoke"
    extra_args+=(--max_steps "$MAX_STEPS" --save_steps "$MAX_STEPS")
elif [[ -n "${MAX_STEPS:-}" ]]; then
    extra_args+=(--max_steps "$MAX_STEPS")
fi

OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME$OUTPUT_DIR_SUFFIX}

cd "$CODE_DIR"

if [[ "$REBUILD_SAMIX" == "1" || ! -f "$SAMIX_JSON" ]]; then
    "$PYTHON_BIN" scripts/v1_5/tools/build_samix_dataset.py \
        --iconqa-json "$ICONQA_JSON" \
        --coco-json "$COCO_JSON" \
        --output-json "$SAMIX_JSON" \
        --manifest-json "$SAMIX_MANIFEST" \
        --data-root "$IMAGE_FOLDER_OVERRIDE" \
        --coco-samples "$COCO_SAMPLES" \
        --seed "$SAMIX_SEED" \
        --shuffle-seed "$SAMIX_SHUFFLE_SEED" \
        --check-images
fi

export STEP_THRESHOLD
export AB_PRESERVE_RATIO
export CMR_LAMBDA
export OMEGA

echo "[SAN-LoRA] data_path=$SAMIX_JSON"
echo "[SAN-LoRA] manifest=$SAMIX_MANIFEST"
echo "[SAN-LoRA] image_folder=$IMAGE_FOLDER_OVERRIDE"
echo "[SAN-LoRA] output_dir=$OUTPUT_DIR"
echo "[SAN-LoRA] trainer=$TRAINER_NAME rank=$LORA_RANK alpha=$LORA_ALPHA"
echo "[SAN-LoRA] coco_samples=$COCO_SAMPLES seed=$SAMIX_SEED shuffle_seed=$SAMIX_SHUFFLE_SEED"
echo "[SAN-LoRA] lr=$LEARNING_RATE lambda=$SAN_LAMBDA warmup=$SAN_WARMUP_STEPS scope=$SAN_SCOPE eps=$SAN_EPS"
echo "[SAN-LoRA] step_threshold=$STEP_THRESHOLD extra_args=${extra_args[*]:-}"

"$DEEPSPEED_BIN" --include "$DEVICE" --master_port "$MASTER_PORT" llava/train/train_mem.py \
    --lora_enable True --lora_r "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --mm_projector_lr 2e-5 \
    --deepspeed "$DEEPSPEED_ZEROFILE" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --version v1 \
    --data_path "$SAMIX_JSON" \
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
    --save_total_limit 15 \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps "$LOGGING_STEPS" \
    --tf32 True \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --lazy_preprocess True \
    --report_to none \
    --trainer_name "$TRAINER_NAME" \
    --san_lambda "$SAN_LAMBDA" \
    --san_warmup_steps "$SAN_WARMUP_STEPS" \
    --san_scope "$SAN_SCOPE" \
    --san_log_every "$SAN_LOG_EVERY" \
    --san_eps "$SAN_EPS" \
    "${extra_args[@]}"
