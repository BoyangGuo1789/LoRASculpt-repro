#!/bin/bash
set -euo pipefail

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

source ./scripts/v1_5/train/trainconfig_migdis_lora.sh

export DEVICE=${DEVICE:-localhost:4,5,6,7}
export MASTER_PORT=${MASTER_PORT:-29640}
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
export TRAINER_NAME="LoRASculptMIGDIS"
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"/data/guoboyang/LoRa-Projects/LoRASculpt-repro/models/llava-v1.5-7b-ft"}
export VISION_TOWER=${VISION_TOWER:-"/data/guoboyang/LoRa-Projects/LoRASculpt-repro/models/clip-vit-large-patch14-336"}
export DATA_PATH_OVERRIDE="/data/guoboyang/LoRa-Projects/LoRASculpt-repro/downloads/official_issue2/extracted/LoRASculpt_JSON_files/IconQA_txt/iconqa_txt-train.json"
export IMAGE_FOLDER_OVERRIDE="/data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/iconqa/iconqa_data"

OUTPUT_DIR_SUFFIX=""
extra_args=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    export STEP_THRESHOLD=${STEP_THRESHOLD_OVERRIDE:-1}
    : ${MAX_STEPS:=$SMOKE_MAX_STEPS}
    OUTPUT_DIR_SUFFIX="-smoke"
    extra_args+=(--max_steps "$MAX_STEPS" --save_steps "$MAX_STEPS")
elif [[ -n "${MAX_STEPS:-}" ]]; then
    extra_args+=(--max_steps "$MAX_STEPS")
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-"/data/guoboyang/LoRa-Projects/LoRASculpt-repro/checkpoints"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OUTPUT_ROOT}/${RUN_NAME}${OUTPUT_DIR_SUFFIX}"}

echo "[LoRASculpt-MIG-DIS] data_path=$DATA_PATH_OVERRIDE"
echo "[LoRASculpt-MIG-DIS] image_folder=$IMAGE_FOLDER_OVERRIDE"
echo "[LoRASculpt-MIG-DIS] output_dir=$OUTPUT_DIR"
echo "[LoRASculpt-MIG-DIS] trainer=$TRAINER_NAME rank=$LORA_RANK alpha=$LORA_ALPHA"
echo "[LoRASculpt-MIG-DIS] grad_mix=$MIGDIS_GRAD_MIX source_margin=$MIGDIS_SOURCE_MARGIN source_scope=$MIGDIS_SOURCE_SCOPE beta=$MIGDIS_GRAD_EMA_BETA norm=$MIGDIS_NORM"
echo "[LoRASculpt-MIG-DIS] selection_mode=$MIGDIS_SELECTION_MODE tgsr_candidate_ratio=$MIGDIS_TGSR_CANDIDATE_RATIO tgsr_core_source_margin=$MIGDIS_TGSR_CORE_SOURCE_MARGIN"
echo "[LoRASculpt-MIG-DIS] dqss_rho=$MIGDIS_DQSS_RHO dqss_aux_grad_mix=$MIGDIS_DQSS_AUX_GRAD_MIX dqss_aux_source_margin=$MIGDIS_DQSS_AUX_SOURCE_MARGIN dqss_module_scope=$MIGDIS_DQSS_MODULE_SCOPE"
echo "[LoRASculpt-MIG-DIS] dqss_anti_collapse=$MIGDIS_DQSS_ANTI_COLLAPSE dqss_max_aux_overlap=$MIGDIS_DQSS_MAX_AUX_OVERLAP dqss_min_core_overlap=$MIGDIS_DQSS_MIN_CORE_OVERLAP"
echo "[LoRASculpt-MIG-DIS] step_threshold=$STEP_THRESHOLD extra_args=${extra_args[*]:-}"

"$DEEPSPEED_BIN" --include "$DEVICE" --master_port "$MASTER_PORT" llava/train/train_mem.py \
    --lora_enable True --lora_r "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --mm_projector_lr 2e-5 \
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
    --migdis_norm_q_low "$MIGDIS_NORM_Q_LOW" \
    --migdis_norm_q_high "$MIGDIS_NORM_Q_HIGH" \
    --migdis_eps "$MIGDIS_EPS" \
    --migdis_source_chunk_rows "$MIGDIS_SOURCE_CHUNK_ROWS" \
    --migdis_final_gamma "$MIGDIS_FINAL_GAMMA" \
    --migdis_debug_dump "$MIGDIS_DEBUG_DUMP" \
    --migdis_selection_mode "$MIGDIS_SELECTION_MODE" \
    --migdis_tgsr_candidate_ratio "$MIGDIS_TGSR_CANDIDATE_RATIO" \
    --migdis_tgsr_core_source_margin "$MIGDIS_TGSR_CORE_SOURCE_MARGIN" \
    --migdis_tgsr_debug_overlap "$MIGDIS_TGSR_DEBUG_OVERLAP" \
    --migdis_dqss_aux_grad_mix "$MIGDIS_DQSS_AUX_GRAD_MIX" \
    --migdis_dqss_aux_source_margin "$MIGDIS_DQSS_AUX_SOURCE_MARGIN" \
    --migdis_dqss_rho "$MIGDIS_DQSS_RHO" \
    --migdis_dqss_debug_overlap "$MIGDIS_DQSS_DEBUG_OVERLAP" \
    --migdis_dqss_module_scope "$MIGDIS_DQSS_MODULE_SCOPE" \
    --migdis_dqss_anti_collapse "$MIGDIS_DQSS_ANTI_COLLAPSE" \
    --migdis_dqss_max_aux_overlap "$MIGDIS_DQSS_MAX_AUX_OVERLAP" \
    --migdis_dqss_min_core_overlap "$MIGDIS_DQSS_MIN_CORE_OVERLAP" \
    "${extra_args[@]}"
