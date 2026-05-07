#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/data/guoboyang/LoRa-Projects/LoRASculpt-repro}
PYTHON_BIN=${PYTHON_BIN:-/data/guoboyang/miniconda3/envs/lorasculpt/bin/python}
TARGET_CHECKPOINT=${TARGET_CHECKPOINT:-$ROOT/checkpoints/llava-v1.5-7b-iconqa_txt_official_issue2-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1-gamma090}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-$ROOT/models/llava-v1.5-7b-ft}
TARGET_MODEL_BASE=${TARGET_MODEL_BASE:-$ROOT/models/llava-v1.5-7b-ft}
SOURCE_MODEL_BASE=${SOURCE_MODEL_BASE:-none}
RUN_NAME=${RUN_NAME:-taskaware_gamma090_target_base_source}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/repro_results/task_aware_gamma}
LOG_DIR=${LOG_DIR:-$ROOT/logs}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
EVAL_SELECTED=${EVAL_SELECTED:-scripts/v1_5/eval/eval_selected_official_issue2_iconqa.sh}

while [ $# -gt 0 ]; do
  case "$1" in
    --target-checkpoint) TARGET_CHECKPOINT="$2"; shift 2 ;;
    --source-checkpoint) SOURCE_CHECKPOINT="$2"; shift 2 ;;
    --target-model-base) TARGET_MODEL_BASE="$2"; shift 2 ;;
    --source-model-base) SOURCE_MODEL_BASE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_ROOT/$RUN_NAME" "$LOG_DIR"
TARGET_RUN="${RUN_NAME}_target_iconqa"
SOURCE_RUN="${RUN_NAME}_source_general"
TARGET_LOG="$LOG_DIR/${TARGET_RUN}.log"
SOURCE_LOG="$LOG_DIR/${SOURCE_RUN}.log"

echo "[task-aware-gate] target IconQA checkpoint: $TARGET_CHECKPOINT"
echo "[task-aware-gate] source/general checkpoint: $SOURCE_CHECKPOINT"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" MODEL_BASE="$TARGET_MODEL_BASE" \
  bash "$EVAL_SELECTED" \
    --checkpoint "$TARGET_CHECKPOINT" \
    --run-name "$TARGET_RUN" \
    --tasks iconqa \
    --output-root "$OUTPUT_ROOT" \
    --log-file "$TARGET_LOG" 2>&1 | tee "$TARGET_LOG"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" MODEL_BASE="$SOURCE_MODEL_BASE" \
  bash "$EVAL_SELECTED" \
    --checkpoint "$SOURCE_CHECKPOINT" \
    --run-name "$SOURCE_RUN" \
    --tasks okvqa,ocrvqa,gqa,textvqa \
    --output-root "$OUTPUT_ROOT" \
    --log-file "$SOURCE_LOG" 2>&1 | tee "$SOURCE_LOG"

"$PYTHON_BIN" - "$RUN_NAME" "$TARGET_CHECKPOINT" "$SOURCE_CHECKPOINT" \
  "$OUTPUT_ROOT/$TARGET_RUN/metrics.json" "$OUTPUT_ROOT/$SOURCE_RUN/metrics.json" \
  "$OUTPUT_ROOT/$RUN_NAME/combined_metrics.json" "$OUTPUT_ROOT/$RUN_NAME/combined_metrics.md" <<\PYCOMBINE
import json
import subprocess
import sys

run_name, target_ckpt, source_ckpt, target_metrics_path, source_metrics_path, out_json, out_md = sys.argv[1:]
with open(target_metrics_path) as f:
    target_metrics = json.load(f)
with open(source_metrics_path) as f:
    source_metrics = json.load(f)
vals = {}
vals.update(target_metrics["metrics"])
for key, value in source_metrics["metrics"].items():
    if value is not None:
        vals[key] = value
source_keys = ["OKVQA", "OCRVQA", "GQA", "TextVQA"]
source_avg = sum(vals[k] for k in source_keys) / len(source_keys)
avg = (vals["IconQA"] + source_avg) / 2
exact_baseline = 70.05375
plus1_target = 71.05375
current_best_static = 70.09625
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    commit = ""
out = {
    "run_name": run_name,
    "method": "task_aware_lora_gate",
    "target_policy": "Use target LoRA checkpoint for IconQA.",
    "source_policy": "Use base LLaVA checkpoint for OKVQA/OCRVQA/GQA/TextVQA.",
    "target_checkpoint": target_ckpt,
    "source_checkpoint": source_ckpt,
    "metrics": {**vals, "SourceAvg": source_avg, "Avg": avg},
    "exact_baseline_avg": exact_baseline,
    "plus1_target_avg": plus1_target,
    "current_best_static_avg": current_best_static,
    "delta_vs_exact_baseline": avg - exact_baseline,
    "delta_vs_plus1_target": avg - plus1_target,
    "delta_vs_current_best_static": avg - current_best_static,
    "passes_plus1_target": avg >= plus1_target,
    "static_single_checkpoint": False,
    "protocol_note": "This is a task-aware inference policy, not a static single-checkpoint LoRASculpt model.",
    "git_commit": commit,
    "component_metrics": {
        "target": target_metrics_path,
        "source": source_metrics_path,
    },
}
with open(out_json, "w") as f:
    json.dump(out, f, indent=2)
with open(out_md, "w") as f:
    f.write(f"# {run_name}\n\n")
    f.write("| Metric | Value |\n|---|---:|\n")
    for key in ["IconQA", "OKVQA", "OCRVQA", "GQA", "TextVQA", "SourceAvg", "Avg"]:
        f.write("| {} | {:.4f} |\n".format(key, out["metrics"][key]))
    f.write("\nDelta vs exact baseline: {:.4f}\n".format(out["delta_vs_exact_baseline"]))
    f.write("\nDelta vs plus-one target: {:.4f}\n".format(out["delta_vs_plus1_target"]))
    f.write("\nProtocol note: task-aware inference policy, not a static single checkpoint.\n")
print(json.dumps(out, indent=2))
PYCOMBINE
