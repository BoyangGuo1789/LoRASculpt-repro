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
MASTER_PORT=${MASTER_PORT:-29720}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
DEEPSPEED_ZEROFILE=${DEEPSPEED_ZEROFILE:-./scripts/zero2.json}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-${EPOCHS:-1}}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
AB_PRESERVE_RATIO=${AB_PRESERVE_RATIO:-0.1}
OMEGA=${OMEGA:-1.0}
CMR_LAMBDA=${CMR_LAMBDA:-1e-3}
LEARNING_RATE=${LEARNING_RATE:-${LR:-5e-5}}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
LOGGING_STEPS=${LOGGING_STEPS:-1}
STEP_THRESHOLD=${STEP_THRESHOLD:-1000000}

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
BASELINE_LORA=${BASELINE_LORA:-$ROOT/checkpoints/llava-v1.5-7b-iconqa_txt_official_issue2-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1}
TEACHER_LORA=${TEACHER_LORA:-$BASELINE_LORA}
SOURCE_WEIGHT=${SOURCE_WEIGHT:-0.50}
LAMBDA_TARGET_KL=${LAMBDA_TARGET_KL:-1.00}
LAMBDA_SOURCE_KL=${LAMBDA_SOURCE_KL:-0.05}
LAMBDA_L2=${LAMBDA_L2:-1e-4}
KL_TEMPERATURE=${KL_TEMPERATURE:-2.0}
KL_TOPK=${KL_TOPK:-64}
USE_PCGRAD=${USE_PCGRAD:-0}
if [[ "$USE_PCGRAD" == "1" || "$USE_PCGRAD" == "true" || "$USE_PCGRAD" == "True" ]]; then
    USE_PCGRAD_ARG=True
else
    USE_PCGRAD_ARG=False
fi

TRAINER_NAME=${TRAINER_NAME:-TPSAMIX}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/checkpoints}
RUN_NAME=${RUN_NAME:-llava-v1.5-7b-lorasculpt-tpsamix-coco${COCO_SAMPLES}-sw${SOURCE_WEIGHT}-tkl${LAMBDA_TARGET_KL}-r${LORA_RANK}}
OUTPUT_DIR_SUFFIX=""
extra_args=()

if [[ "${SMOKE:-0}" == "1" ]]; then
    STEP_THRESHOLD=${STEP_THRESHOLD_OVERRIDE:-1000000}
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

echo "[TP-SA-MIX] data_path=$SAMIX_JSON"
echo "[TP-SA-MIX] manifest=$SAMIX_MANIFEST"
echo "[TP-SA-MIX] image_folder=$IMAGE_FOLDER_OVERRIDE"
echo "[TP-SA-MIX] output_dir=$OUTPUT_DIR"
echo "[TP-SA-MIX] baseline_lora=$BASELINE_LORA"
echo "[TP-SA-MIX] teacher_lora=$TEACHER_LORA"
echo "[TP-SA-MIX] trainer=$TRAINER_NAME rank=$LORA_RANK alpha=$LORA_ALPHA"
echo "[TP-SA-MIX] coco_samples=$COCO_SAMPLES seed=$SAMIX_SEED shuffle_seed=$SAMIX_SHUFFLE_SEED"
echo "[TP-SA-MIX] lr=$LEARNING_RATE source_weight=$SOURCE_WEIGHT target_kl=$LAMBDA_TARGET_KL source_kl=$LAMBDA_SOURCE_KL l2=$LAMBDA_L2 topk=$KL_TOPK pcgrad=$USE_PCGRAD"
echo "[TP-SA-MIX] step_threshold=$STEP_THRESHOLD extra_args=${extra_args[*]:-}"

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
    --lora_start_path "$BASELINE_LORA" \
    --tp_samix_teacher_lora_path "$TEACHER_LORA" \
    --tp_samix_source_weight "$SOURCE_WEIGHT" \
    --tp_samix_lambda_target_kl "$LAMBDA_TARGET_KL" \
    --tp_samix_lambda_source_kl "$LAMBDA_SOURCE_KL" \
    --tp_samix_lambda_l2 "$LAMBDA_L2" \
    --tp_samix_kl_temperature "$KL_TEMPERATURE" \
    --tp_samix_kl_topk "$KL_TOPK" \
    --tp_samix_use_pcgrad "$USE_PCGRAD_ARG" \
    "${extra_args[@]}"
