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

DEVICE=${DEVICE:-localhost:0,1,3,4}
MASTER_PORT=${MASTER_PORT:-29780}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
FREEZE_RANK=${FREEZE_RANK:-32}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
MM_PROJECTOR_LR=${MM_PROJECTOR_LR:-0}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
DEEPSPEED_ZEROFILE=${DEEPSPEED_ZEROFILE:-./scripts/zero2.json}

OFFICIAL_JSON_ROOT=${OFFICIAL_JSON_ROOT:-$ROOT/downloads/official_issue2/extracted/LoRASculpt_JSON_files}
COCO_TARGET_JSON=${COCO_TARGET_JSON:-$OFFICIAL_JSON_ROOT/COCO-Caption/coco-train.json}
OKVQA_JSON=${OKVQA_JSON:-$ROOT/data/source_adapter/okvqa_train_all_seed42.json}
OKVQA_SAMPLES=${OKVQA_SAMPLES:-3000}
TFR_SEED=${TFR_SEED:-42}
TFR_DATA_DIR=${TFR_DATA_DIR:-$ROOT/data/tfr_lora}
MIX_JSON=${MIX_JSON:-$TFR_DATA_DIR/coco_target_okvqa_s${OKVQA_SAMPLES}_seed${TFR_SEED}.json}
MIX_MANIFEST=${MIX_MANIFEST:-${MIX_JSON%.json}.manifest.json}
REBUILD_MIX=${REBUILD_MIX:-0}

BASELINE_LORA=${BASELINE_LORA:-$ROOT/checkpoints/llava-v1.5-7b-coco_official_issue2-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1}
INIT_LORA=${INIT_LORA:-$ROOT/checkpoints/llava-v1.5-7b-coco_official_issue2-lorasculpt-tfr-init-target-r${LORA_RANK}-a${LORA_ALPHA}}
LOAD_NON_LORA=${LOAD_NON_LORA:-True}

SOURCE_WEIGHT=${SOURCE_WEIGHT:-1.0}
LAMBDA_TARGET_KL=${LAMBDA_TARGET_KL:-0.0}
LAMBDA_SOURCE_KL=${LAMBDA_SOURCE_KL:-0.0}
KL_TEMPERATURE=${KL_TEMPERATURE:-2.0}
KL_TOPK=${KL_TOPK:-64}
USE_PCGRAD=${USE_PCGRAD:-0}
RESIDUAL_L2=${RESIDUAL_L2:-0.0}

TRAINER_NAME=${TRAINER_NAME:-TFRLORA}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/checkpoints}
RUN_NAME=${RUN_NAME:-llava-v1.5-7b-lorasculpt-tfr-bs-coco-okvqa-s${OKVQA_SAMPLES}-sw${SOURCE_WEIGHT}-tkl${LAMBDA_TARGET_KL}-r${LORA_RANK}}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}
SMOKE=${SMOKE:-0}
SMOKE_MAX_STEPS=${SMOKE_MAX_STEPS:-20}
SAVE_STEPS=${SAVE_STEPS:-10000}

if [[ "$USE_PCGRAD" == "1" || "$USE_PCGRAD" == "true" || "$USE_PCGRAD" == "True" ]]; then
    USE_PCGRAD_ARG=True
else
    USE_PCGRAD_ARG=False
fi

cd "$CODE_DIR"

if [ ! -f "$OKVQA_JSON" ]; then
    "$PYTHON_BIN" scripts/v1_5/train/build_okvqa_train_lora_json.py \
        --questions "$ROOT/data/okvqa/OpenEnded_mscoco_train2014_questions.json" \
        --annotations "$ROOT/data/okvqa/mscoco_train2014_annotations.json" \
        --image-root "$ROOT/data/coco/train2014" \
        --output "$OKVQA_JSON" \
        --sample-size 0 \
        --seed "$TFR_SEED" \
        --skip-missing-images
fi

if [[ "$REBUILD_MIX" == "1" || ! -f "$MIX_JSON" ]]; then
    "$PYTHON_BIN" scripts/v1_5/tools/build_coco_target_okvqa_mix_dataset.py \
        --coco-target-json "$COCO_TARGET_JSON" \
        --okvqa-json "$OKVQA_JSON" \
        --output-json "$MIX_JSON" \
        --manifest-json "$MIX_MANIFEST" \
        --data-root "$ROOT/data" \
        --okvqa-samples "$OKVQA_SAMPLES" \
        --seed "$TFR_SEED" \
        --shuffle-seed "$TFR_SEED" \
        --check-images
fi

if [ ! -f "$INIT_LORA/adapter_model.bin" ]; then
    "$PYTHON_BIN" scripts/v1_5/tools/rank_expanded_residual_lora.py \
        --target-checkpoint "$BASELINE_LORA" \
        --source-checkpoint "$BASELINE_LORA" \
        --output-checkpoint "$INIT_LORA" \
        --scope all \
        --layer-band all \
        --source-lambda 0.0 \
        --output-rank "$LORA_RANK" \
        --output-alpha "$LORA_ALPHA" \
        --overwrite
fi

extra_args=(--save_steps "$SAVE_STEPS")
if [ "$SMOKE" = "1" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}-${TS}-smoke"
    extra_args=(--max_steps "$SMOKE_MAX_STEPS" --save_steps "$SMOKE_MAX_STEPS")
fi

echo "[TFR-LoRA COCO] data_path=$MIX_JSON"
echo "[TFR-LoRA COCO] manifest=$MIX_MANIFEST"
echo "[TFR-LoRA COCO] image_folder=$ROOT/data"
echo "[TFR-LoRA COCO] output_dir=$OUTPUT_DIR"
echo "[TFR-LoRA COCO] baseline_lora=$BASELINE_LORA"
echo "[TFR-LoRA COCO] init_lora=$INIT_LORA"
echo "[TFR-LoRA COCO] trainer=$TRAINER_NAME rank=$LORA_RANK alpha=$LORA_ALPHA freeze_rank=$FREEZE_RANK"
echo "[TFR-LoRA COCO] okvqa_samples=$OKVQA_SAMPLES seed=$TFR_SEED"
echo "[TFR-LoRA COCO] lr=$LEARNING_RATE mm_projector_lr=$MM_PROJECTOR_LR source_weight=$SOURCE_WEIGHT target_kl=$LAMBDA_TARGET_KL source_kl=$LAMBDA_SOURCE_KL residual_l2=$RESIDUAL_L2 pcgrad=$USE_PCGRAD_ARG"

"$DEEPSPEED_BIN" --include "$DEVICE" --master_port "$MASTER_PORT" llava/train/train_mem.py \
    --lora_enable True --lora_r "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --mm_projector_lr "$MM_PROJECTOR_LR" \
    --deepspeed "$DEEPSPEED_ZEROFILE" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --version v1 \
    --data_path "$MIX_JSON" \
    --image_folder "$ROOT/data" \
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
    --save_total_limit 15 \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --lazy_preprocess True \
    --report_to none \
    --trainer_name "$TRAINER_NAME" \
    --lora_start_path "$INIT_LORA" \
    --load_lora_start_non_lora "$LOAD_NON_LORA" \
    --tp_samix_teacher_lora_path "$INIT_LORA" \
    --tp_samix_source_weight "$SOURCE_WEIGHT" \
    --tp_samix_lambda_target_kl "$LAMBDA_TARGET_KL" \
    --tp_samix_lambda_source_kl "$LAMBDA_SOURCE_KL" \
    --tp_samix_kl_temperature "$KL_TEMPERATURE" \
    --tp_samix_kl_topk "$KL_TOPK" \
    --tp_samix_use_pcgrad "$USE_PCGRAD_ARG" \
    --tfr_freeze_rank "$FREEZE_RANK" \
    --tfr_residual_l2 "$RESIDUAL_L2" \
    "${extra_args[@]}"
