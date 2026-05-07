#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/data/guoboyang/LoRa-Projects/LoRASculpt-repro}
CODE_DIR=${CODE_DIR:-$ROOT/LoRASculpt}
PYTHON_BIN=${PYTHON_BIN:-/data/guoboyang/miniconda3/envs/lorasculpt/bin/python}

BASE_CKPT=${BASE_CKPT:-$ROOT/checkpoints/llava-v1.5-7b-iconqa_txt_official_issue2-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1}
SAMIX1500_CKPT=${SAMIX1500_CKPT:-$ROOT/checkpoints/llava-v1.5-7b-lorasculpt-samix-coco1500-iconqa-r32}
SAMIX3000_CKPT=${SAMIX3000_CKPT:-$ROOT/checkpoints/llava-v1.5-7b-lorasculpt-samix-coco3000-iconqa-r32}
MANIFEST=${MANIFEST:-$CODE_DIR/scripts/v1_5/posthoc/samix_delta_fusion_manifest.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/checkpoints}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-llava-v1.5-7b-lorasculpt-samixdelta}
RANK=${RANK:-32}
ALPHA=${ALPHA:-64}

if [ "$#" -gt 0 ]; then
  CANDIDATES=("$@")
else
  CANDIDATES=(
    samix1500_v_mid_l005
    samix1500_vo_mid_l005
    samix3000_v_mid_l005
    samix1500_mlp_mid_l005
  )
fi

cd "$CODE_DIR"

for CANDIDATE in "${CANDIDATES[@]}"; do
  OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_PREFIX-$CANDIDATE"
  echo "[samix_delta_fusion] candidate=$CANDIDATE"
  echo "[samix_delta_fusion] output_dir=$OUTPUT_DIR"
  "$PYTHON_BIN" scripts/v1_5/posthoc/fuse_lora_delta_basis.py \
    --base_ckpt "$BASE_CKPT" \
    --basis_ckpts "samix1500=$SAMIX1500_CKPT,samix3000=$SAMIX3000_CKPT" \
    --manifest "$MANIFEST" \
    --candidate "$CANDIDATE" \
    --output_dir "$OUTPUT_DIR" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --write_meta \
    --overwrite
done
