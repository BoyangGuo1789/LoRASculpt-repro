#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/data/guoboyang/LoRa-Projects/LoRASculpt-repro}
CODE_DIR=${CODE_DIR:-$ROOT/LoRASculpt}
PYTHON_BIN=${PYTHON_BIN:-/data/guoboyang/miniconda3/envs/lorasculpt/bin/python}

BASE_CKPT=${BASE_CKPT:-$ROOT/checkpoints/llava-v1.5-7b-iconqa_txt_official_issue2-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1-gamma090}
OKVQA_CKPT=${OKVQA_CKPT:?Set OKVQA_CKPT to a source OKVQA LoRA checkpoint}
MANIFEST=${MANIFEST:-$CODE_DIR/scripts/v1_5/posthoc/okvqa_delta_fusion_manifest.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/checkpoints}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-llava-v1.5-7b-lorasculpt-okvqadelta}
RANK=${RANK:-32}
ALPHA=${ALPHA:-64}

if [ "$#" -gt 0 ]; then
  CANDIDATES=("$@")
else
  CANDIDATES=(
    okvqa_v_mid_l002
    okvqa_qv_mid_l002
    okvqa_v_mid_l005
    okvqa_qv_mid_l005
    okvqa_qkvo_mid_l002
  )
fi

cd "$CODE_DIR"

for CANDIDATE in "${CANDIDATES[@]}"; do
  OUTPUT_DIR="$OUTPUT_ROOT/$OUTPUT_PREFIX-$CANDIDATE"
  echo "[okvqa_delta_fusion] candidate=$CANDIDATE"
  echo "[okvqa_delta_fusion] base_ckpt=$BASE_CKPT"
  echo "[okvqa_delta_fusion] okvqa_ckpt=$OKVQA_CKPT"
  echo "[okvqa_delta_fusion] output_dir=$OUTPUT_DIR"
  "$PYTHON_BIN" scripts/v1_5/posthoc/fuse_lora_delta_basis.py \
    --base_ckpt "$BASE_CKPT" \
    --basis_ckpts "okvqa=$OKVQA_CKPT" \
    --manifest "$MANIFEST" \
    --candidate "$CANDIDATE" \
    --output_dir "$OUTPUT_DIR" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --write_meta \
    --overwrite
done
