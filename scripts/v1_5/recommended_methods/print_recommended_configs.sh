#!/bin/bash
set -euo pipefail

echo "# Recommended Method Configs"
echo "Generated: $(date -Is)"
echo
cat docs/recommended_methods/SPIDER_source_audit.md
echo
cat docs/recommended_methods/AdaDARE_gamma_source_audit.md
echo
cat <<CFG
# Runnable defaults

SPIDER:
  script: scripts/v1_5/train/trainconfig_spider.sh
  trainer_name: SPIDER
  lora_enable: false
  tune_mm_mlp_adapter: true
  tune_decoder_layer: ${TUNE_DECODER_LAYER:-2}
  learning_rate: ${LEARNING_RATE:-2e-4}
  mm_projector_lr: ${MM_PROJECTOR_LR:-2e-5}
  epochs: ${NUM_TRAIN_EPOCHS:-5}

AdaDARE-gamma:
  train_script: scripts/v1_5/train/trainconfig_adadare_lora.sh
  fuse_script: scripts/v1_5/recommended_methods/adadare_gamma_fuse.py
  trainer_name: LLaVATrainer
  lora_rank: ${LORA_RANK:-128}
  lora_alpha: ${LORA_ALPHA:-256}
  learning_rate: ${LEARNING_RATE:-5e-6}
  epochs: ${NUM_TRAIN_EPOCHS:-5}
  gamma: ${ADADARE_GAMMA:-0.7}
  sparsity: ${ADADARE_SPARSITY:-0.9}

Evaluation:
  script: scripts/v1_5/eval/recommended/eval_recommended_method.sh
  targets: iconqa_txt, coco
  source_eval: OKVQA, OCRVQA, GQA, TextVQA
CFG
