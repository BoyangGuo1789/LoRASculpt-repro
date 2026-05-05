#!/bin/bash
set -euo pipefail

CKPT_PATH=${1:-/data/guoboyang/LoRa-Projects/LoRASculpt-repro/checkpoints/llava-v1.5-7b-lorasculpt-dis-iconqa-r32}
RUN_TAG=${RUN_TAG:-$(basename "$CKPT_PATH")}
export RESULT_ROOT=${RESULT_ROOT:-/data/guoboyang/LoRa-Projects/LoRASculpt-repro/repro_results/20260505_lorasculpt_dis_iconqa_official_issue2_rank32_${RUN_TAG}}
export SUMMARY_OUTPUT_DIR=${SUMMARY_OUTPUT_DIR:-$RESULT_ROOT/summary.txt}
export PYTHON_BIN=${PYTHON_BIN:-/data/guoboyang/miniconda3/envs/lorasculpt/bin/python}

echo "[LoRASculpt-DIS eval] ckpt=$CKPT_PATH"
echo "[LoRASculpt-DIS eval] result_root=$RESULT_ROOT"

bash scripts/v1_5/eval/eval_all_official_issue2_iconqa.sh "$CKPT_PATH"
