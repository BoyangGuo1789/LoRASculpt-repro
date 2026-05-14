#!/bin/bash
set -euo pipefail

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

source ./scripts/v1_5/train/trainconfig_pars_lora.sh

export DEVICE=${DEVICE:-localhost:0,1,2,3}
export MASTER_PORT=${MASTER_PORT:-29660}
export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
export DEEPSPEED_ZEROFILE=${DEEPSPEED_ZEROFILE:-"./scripts/zero2.json"}
export DEEPSPEED_BIN=${DEEPSPEED_BIN:-"/data/guoboyang/miniconda3/envs/lorasculpt/bin/deepspeed"}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export AB_PRESERVE_RATIO=${AB_PRESERVE_RATIO:-0.1}
export OMEGA=${OMEGA:-1.0}
export CMR_LAMBDA=${CMR_LAMBDA:-1e-3}
export STEP_THRESHOLD=${STEP_THRESHOLD:-100}

export DATASET_NAME="iconqa_txt"
export TRAINER_NAME="LoRASculptPARS"
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"/data/guoboyang/LoRa-Projects/LoRASculpt-repro/models/llava-v1.5-7b-ft"}
export VISION_TOWER=${VISION_TOWER:-"/data/guoboyang/LoRa-Projects/LoRASculpt-repro/models/clip-vit-large-patch14-336"}
export DATA_PATH_OVERRIDE="/data/guoboyang/LoRa-Projects/LoRASculpt-repro/downloads/official_issue2/extracted/LoRASculpt_JSON_files/IconQA_txt/iconqa_txt-train.json"
export IMAGE_FOLDER_OVERRIDE="/data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/iconqa/iconqa_data"

OUTPUT_DIR_SUFFIX=""
extra_args=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    export STEP_THRESHOLD=${STEP_THRESHOLD_OVERRIDE:-1}
    : ${SMOKE_MAX_STEPS:=20}
    : ${MAX_STEPS:=$SMOKE_MAX_STEPS}
    OUTPUT_DIR_SUFFIX="-smoke"
    extra_args+=(--max_steps "$MAX_STEPS" --save_steps "$MAX_STEPS")
elif [[ -n "${MAX_STEPS:-}" ]]; then
    extra_args+=(--max_steps "$MAX_STEPS")
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-"/data/guoboyang/LoRa-Projects/LoRASculpt-repro/checkpoints"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${RUN_NAME}${OUTPUT_DIR_SUFFIX}"}

echo "[PARS-LoRA] data_path=$DATA_PATH_OVERRIDE"
echo "[PARS-LoRA] image_folder=$IMAGE_FOLDER_OVERRIDE"
echo "[PARS-LoRA] output_dir=$OUTPUT_DIR"
echo "[PARS-LoRA] trainer=$TRAINER_NAME rank=$LORA_RANK alpha=$LORA_ALPHA ratio=$AB_PRESERVE_RATIO cmr_lambda=$CMR_LAMBDA omega=$OMEGA step_threshold=$STEP_THRESHOLD"
echo "[PARS-LoRA] projector_lambda=$PARS_PROJECTOR_LAMBDA projector_tau=$PARS_PROJECTOR_TAU projector_warmup=$PARS_PROJECTOR_WARMUP_STEPS"
echo "[PARS-LoRA] stable_rank=$PARS_STABLE_RANK stable_lr_mult=$PARS_STABLE_LR_MULT orth_lambda=$PARS_ORTH_LAMBDA"
echo "[PARS-LoRA] migdis_enable=$MIGDIS_ENABLE migdis_selection_mode=$MIGDIS_SELECTION_MODE"
echo "[PARS-LoRA] extra_args=${extra_args[*]:-}"

"$DEEPSPEED_BIN" --include "$DEVICE" --master_port "$MASTER_PORT" llava/train/train_mem.py \
    --lora_enable True \
    --lora_r "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --mm_projector_lr "${MM_PROJECTOR_LR:-2e-5}" \
    --deepspeed "$DEEPSPEED_ZEROFILE" \
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
    --save_total_limit 15 \
    --learning_rate "${LEARNING_RATE:-2e-4}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps "${LOGGING_STEPS:-1}" \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH:-2048}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-True}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
    --lazy_preprocess True \
    --report_to none \
    --trainer_name "$TRAINER_NAME" \
    --migdis_enable "$MIGDIS_ENABLE" \
    --migdis_grad_ema_beta "$MIGDIS_GRAD_EMA_BETA" \
    --migdis_grad_mix "$MIGDIS_GRAD_MIX" \
    --migdis_source_margin "$MIGDIS_SOURCE_MARGIN" \
    --migdis_source_scope "$MIGDIS_SOURCE_SCOPE" \
    --migdis_norm "$MIGDIS_NORM" \
    --migdis_eps "$MIGDIS_EPS" \
    --migdis_final_gamma "$MIGDIS_FINAL_GAMMA" \
    --migdis_debug_dump "$MIGDIS_DEBUG_DUMP" \
    --migdis_selection_mode "$MIGDIS_SELECTION_MODE" \
    --pars_enable "$PARS_ENABLE" \
    --pars_projector_lambda "$PARS_PROJECTOR_LAMBDA" \
    --pars_projector_tau "$PARS_PROJECTOR_TAU" \
    --pars_projector_warmup_steps "$PARS_PROJECTOR_WARMUP_STEPS" \
    --pars_stable_rank "$PARS_STABLE_RANK" \
    --pars_stable_lr_mult "$PARS_STABLE_LR_MULT" \
    --pars_orth_lambda "$PARS_ORTH_LAMBDA" \
    --pars_log_every "$PARS_LOG_EVERY" \
    --pars_eps "$PARS_EPS" \
    "${extra_args[@]}"
