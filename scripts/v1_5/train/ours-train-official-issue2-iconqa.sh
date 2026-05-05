#!/bin/bash

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export DEVICE=${DEVICE:-localhost:4,5,6,7}
OUTPUT_DIR_PREFIX="/data/guoboyang/LoRa-Projects/LoRASculpt-repro/checkpoints/llava-v1.5-7b"

export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
export DEEPSPEED_ZEROFILE=${DEEPSPEED_ZEROFILE:-"./scripts/zero2.json"}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export AB_PRESERVE_RATIO=${AB_PRESERVE_RATIO:-0.1}
export OMEGA=${OMEGA:-1.0}
export CMR_LAMBDA=${CMR_LAMBDA:-1e-3}

export DATASET_NAME="iconqa_txt"
export TRAINER_NAME="LoRASculpt"
export DATA_PATH_OVERRIDE="/data/guoboyang/LoRa-Projects/LoRASculpt-repro/downloads/official_issue2/extracted/LoRASculpt_JSON_files/IconQA_txt/iconqa_txt-train.json"
export IMAGE_FOLDER_OVERRIDE="/data/guoboyang/LoRa-Projects/LoRASculpt-repro/data/iconqa/iconqa_data"

HYPERPARAMS="lora-r${LORA_RANK}-a${LORA_ALPHA}-e${NUM_TRAIN_EPOCHS}-CMRLAMBDA${CMR_LAMBDA}-OMEGA${OMEGA}-RATIO${AB_PRESERVE_RATIO}"
export OUTPUT_DIR="${OUTPUT_DIR_PREFIX}-iconqa_txt_official_issue2-${TRAINER_NAME}-${HYPERPARAMS}"

bash ./scripts/v1_5/train/trainconfig_lora.sh
