#!/bin/bash
set -euo pipefail

: "${ROOT:=/data/guoboyang/LoRa-Projects/LoRASculpt-repro}"
: "${REPO:=$ROOT/LoRASculpt}"
: "${CKPTS:=$ROOT/checkpoints}"
: "${LOGS:=$ROOT/logs}"
: "${PYTHON_BIN:=/data/guoboyang/miniconda3/envs/lorasculpt/bin/python}"

cd "$REPO"

BASE_EXACT="${BASE_EXACT:-$CKPTS/llava-v1.5-7b-iconqa_txt_official_issue2-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1}"
BASE_G090="${BASE_G090:-$CKPTS/llava-v1.5-7b-iconqa_txt_official_issue2-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1-gamma090}"
MIGDIS_V4="${MIGDIS_V4:-$CKPTS/llava-v1.5-7b-lorasculpt-migdis-iconqa-r32-g010-sm035-qkv}"
MIGDIS_V5="${MIGDIS_V5:-$CKPTS/llava-v1.5-7b-lorasculpt-migdis-iconqa-r32-g025-sm070-qkv}"
DQSS_R025G="${DQSS_R025G:-$CKPTS/llava-v1.5-7b-lorasculpt-migdis-dqss-r025g-iconqa-r32}"
MANIFEST="${MANIFEST:-scripts/v1_5/posthoc/mbldf_plus1_manifest.json}"

mkdir -p "$LOGS" experiments/mbldf_plus1
touch experiments/mbldf_plus1/results.csv

mapfile -t CANDIDATES < <("$PYTHON_BIN" - <<'PY'
import json
m=json.load(open("scripts/v1_5/posthoc/mbldf_plus1_manifest.json"))
for c in m["candidates"]:
    if c["name"] != "identity_g090":
        print(c["name"])
PY
)

for RUN in "${CANDIDATES[@]}"; do
  OUT="$CKPTS/llava-v1.5-7b-lorasculpt-mbldf-$RUN"
  TS=$(date +%Y%m%d_%H%M%S)
  LOG="$LOGS/mbldf_${RUN}_fuse_${TS}.log"
  "$PYTHON_BIN" scripts/v1_5/posthoc/fuse_lora_delta_basis.py \
    --base_ckpt "$BASE_EXACT" \
    --basis_ckpts "base_g090=$BASE_G090,v5=$MIGDIS_V5,dqss=$DQSS_R025G,v4=$MIGDIS_V4" \
    --manifest "$MANIFEST" \
    --candidate "$RUN" \
    --output_dir "$OUT" \
    --rank 32 \
    --alpha 64 \
    --write_meta \
    --overwrite 2>&1 | tee "$LOG"

  "$PYTHON_BIN" scripts/v1_5/tools/collect_mbldf_results.py \
    --fusion_meta "$OUT/fusion_meta.json" \
    --results_csv experiments/mbldf_plus1/results.csv \
    --stage fused \
    --fusion_log_path "$LOG"
done
