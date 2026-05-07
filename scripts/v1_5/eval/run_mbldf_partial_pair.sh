#!/bin/bash
set -euo pipefail

: "${ROOT:=/data/guoboyang/LoRa-Projects/LoRASculpt-repro}"
: "${REPO:=$ROOT/LoRASculpt}"
: "${CKPTS:=$ROOT/checkpoints}"
: "${RESULTS:=$ROOT/repro_results}"
: "${LOGS:=$ROOT/logs}"
: "${PYTHON_BIN:=/data/guoboyang/miniconda3/envs/lorasculpt/bin/python}"
: "${TASKS:=iconqa,okvqa,ocrvqa,textvqa}"

SCRIPT_PATH="$(readlink -f "$0")"

if [ "${1:-}" = "--child" ]; then
  RUN_NAME="${2:?run name required}"
  GPUS="${3:?gpu list required}"
  LOG_FILE="${4:?log file required}"
  CHECKPOINT="$CKPTS/llava-v1.5-7b-lorasculpt-mbldf-$RUN_NAME"

  if [ ! -d "$CHECKPOINT" ]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 2
  fi

  mkdir -p "$LOGS" "$RESULTS/mbldf_plus1"
  cd "$REPO"
  export CUDA_VISIBLE_DEVICES="$GPUS"
  export PYTHON_BIN
  exec bash scripts/v1_5/eval/eval_selected_official_issue2_iconqa.sh \
    --checkpoint "$CHECKPOINT" \
    --run-name "$RUN_NAME" \
    --tasks "$TASKS" \
    --output-root "$RESULTS/mbldf_plus1" \
    --log-file "$LOG_FILE" > "$LOG_FILE" 2>&1
fi

RUN_A="${1:?usage: $0 RUN_A [RUN_B]}"
RUN_B="${2:-}"
GPUS_A="${GPUS_A:-0,1,2,3}"
GPUS_B="${GPUS_B:-4,5,6,7}"

mkdir -p "$LOGS"

launch_one() {
  local run_name="$1"
  local gpus="$2"
  local ts log_file pid
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="$LOGS/${run_name}_partial_eval_${ts}.log"
  ROOT="$ROOT" REPO="$REPO" CKPTS="$CKPTS" RESULTS="$RESULTS" LOGS="$LOGS" \
    PYTHON_BIN="$PYTHON_BIN" TASKS="$TASKS" \
    nohup bash "$SCRIPT_PATH" --child "$run_name" "$gpus" "$log_file" >/dev/null 2>&1 &
  pid=$!
  echo "$run_name $pid $gpus $log_file"
}

launch_one "$RUN_A" "$GPUS_A"
if [ -n "$RUN_B" ]; then
  launch_one "$RUN_B" "$GPUS_B"
fi
