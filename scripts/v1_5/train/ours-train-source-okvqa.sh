#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/data/guoboyang/LoRa-Projects/LoRASculpt-repro}
CODE_DIR=${CODE_DIR:-$ROOT/LoRASculpt}
DEVICE=${DEVICE:-localhost:0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29620}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
TRAINER_NAME=${TRAINER_NAME:-LLaVATrainer}
PYTHON_BIN=${PYTHON_BIN:-/data/guoboyang/miniconda3/envs/lorasculpt/bin/python}
DEEPSPEED_BIN=${DEEPSPEED_BIN:-/data/guoboyang/miniconda3/envs/lorasculpt/bin/deepspeed}
RUN_NAME=${RUN_NAME:-source-okvqa-r32}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/checkpoints}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/llava-v1.5-7b-lorasculpt-${RUN_NAME}-${TS}}
OKVQA_JSON=${OKVQA_JSON:-$ROOT/data/source_adapter/okvqa_train_all_seed42.json}
OKVQA_SAMPLE_SIZE=${OKVQA_SAMPLE_SIZE:-0}
SMOKE=${SMOKE:-0}
SMOKE_MAX_STEPS=${SMOKE_MAX_STEPS:-20}
SAVE_STEPS=${SAVE_STEPS:-10000}

cd "$CODE_DIR"

if [ ! -f "$OKVQA_JSON" ]; then
  "$PYTHON_BIN" scripts/v1_5/train/build_okvqa_train_lora_json.py \
    --questions "$ROOT/data/okvqa/OpenEnded_mscoco_train2014_questions.json" \
    --annotations "$ROOT/data/okvqa/mscoco_train2014_annotations.json" \
    --image-root "$ROOT/data/coco/train2014" \
    --output "$OKVQA_JSON" \
    --sample-size "$OKVQA_SAMPLE_SIZE" \
    --seed 42 \
    --skip-missing-images
fi

extra_args=()
if [ "$SMOKE" = "1" ]; then
  OUTPUT_DIR="${OUTPUT_DIR}-smoke"
  extra_args+=(--max_steps "$SMOKE_MAX_STEPS" --save_steps "$SMOKE_MAX_STEPS")
else
  extra_args+=(--save_steps "$SAVE_STEPS")
fi

echo "[source-okvqa] data_path=$OKVQA_JSON"
echo "[source-okvqa] image_folder=$ROOT/data/coco/train2014"
echo "[source-okvqa] output_dir=$OUTPUT_DIR"
echo "[source-okvqa] trainer=$TRAINER_NAME lr=$LEARNING_RATE epochs=$NUM_TRAIN_EPOCHS smoke=$SMOKE"

"$DEEPSPEED_BIN" --include "$DEVICE" --master_port "$MASTER_PORT" llava/train/train_mem.py \
    --lora_enable True --lora_r "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "$ROOT/models/llava-v1.5-7b-ft" \
    --version v1 \
    --data_path "$OKVQA_JSON" \
    --image_folder "$ROOT/data/coco/train2014" \
    --vision_tower "$ROOT/models/clip-vit-large-patch14-336" \
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
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --trainer_name "$TRAINER_NAME" \
    "${extra_args[@]}"
