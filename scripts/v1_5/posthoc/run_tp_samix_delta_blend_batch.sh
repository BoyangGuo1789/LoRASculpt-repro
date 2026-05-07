#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/data/guoboyang/LoRa-Projects/LoRASculpt-repro}
CODE_DIR=${CODE_DIR:-$ROOT/LoRASculpt}
PYTHON_BIN=${PYTHON_BIN:-/data/guoboyang/miniconda3/envs/lorasculpt/bin/python}

BASE_CKPT=${BASE_CKPT:-$ROOT/checkpoints/llava-v1.5-7b-iconqa_txt_official_issue2-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1}
TPKD_CKPT=${TPKD_CKPT:-$ROOT/checkpoints/llava-v1.5-7b-lorasculpt-tpsamix-kd-coco3000-safe}
TPSRC_CKPT=${TPSRC_CKPT:-$ROOT/checkpoints/llava-v1.5-7b-lorasculpt-tpsamix-src-coco3000-nopcgrad}
MANIFEST=${MANIFEST:-$CODE_DIR/scripts/v1_5/posthoc/tp_samix_delta_blend_manifest.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/checkpoints}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-llava-v1.5-7b-lorasculpt-tpsamixblend}
RANK=${RANK:-32}
ALPHA=${ALPHA:-64}

if [ "$#" -gt 0 ]; then
  CANDIDATES=("$@")
else
  CANDIDATES=(
    tpsrc_all_l002
    tpsrc_all_l005
  )
fi

cd "$CODE_DIR"

for CANDIDATE in "${CANDIDATES[@]}"; do
  OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_PREFIX-$CANDIDATE"
  echo "[tp_samix_delta_blend] candidate=$CANDIDATE"
  echo "[tp_samix_delta_blend] output_dir=$OUTPUT_DIR"
  "$PYTHON_BIN" scripts/v1_5/posthoc/fuse_lora_delta_basis.py \
    --base_ckpt "$BASE_CKPT" \
    --basis_ckpts "tpkd=$TPKD_CKPT,tpsrc=$TPSRC_CKPT" \
    --manifest "$MANIFEST" \
    --candidate "$CANDIDATE" \
    --output_dir "$OUTPUT_DIR" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --write_meta \
    --overwrite
done
