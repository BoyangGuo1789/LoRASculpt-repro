# Paper Preparation Summary

This directory is the paper-facing index for the LoRASculpt reproduction and follow-up experiments in this fork.

## Current Best Result

The best recorded result is `task_aware_lora_gate`, an inference-time routing policy:

- Use the current best IconQA target LoRA checkpoint for IconQA.
- Use the base LLaVA-1.5-7B checkpoint for OKVQA, OCRVQA, GQA, and TextVQA.

This reaches `Avg=73.6825`, which exceeds the reproduced exact baseline `70.05375` by `+3.62875` and the plus-one target `71.05375` by `+2.62875`.

This is not a static single-checkpoint result. It is a task-aware policy whose validity depends on knowing the evaluation task identity at inference time.

## Main Result Table

| Method | IconQA | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg | Avg | Delta vs Exact Baseline | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Reproduced exact baseline | 86.26 | 52.71 | 54.70 | 56.34 | 51.64 | 53.8475 | 70.05375 | 0.0000 | Static LoRA baseline. |
| Current best static gamma090 | 86.29 | 52.70 | 54.75 | 56.38 | 51.78 | 53.9025 | 70.09625 | +0.0425 | Static post-hoc gamma scaling. |
| Task-aware LoRA gate | 86.29 | 57.99 | 66.15 | 61.93 | 58.23 | 61.0750 | 73.6825 | +3.6288 | Dynamic task-aware policy, not a static checkpoint. |

## Evidence Locations

Primary ledger:

- `experiments/task_aware_gamma/results.csv`
- `experiments/task_aware_gamma/metrics/taskaware_gamma090_target_base_source_20260508.combined.json`
- `experiments/task_aware_gamma/README.md`

Component metrics:

- IconQA target LoRA component: `experiments/task_aware_gamma/metrics/taskaware_gamma090_target_base_source_20260508.target_iconqa.json`
- Base source three-task component: `experiments/task_aware_gamma/metrics/taskaware_gamma090_target_base_source_20260508.source3.json`
- Base GQA component: `experiments/task_aware_gamma/metrics/taskaware_gamma090_target_base_source_20260508.gqa.json`

Execution script:

- `scripts/v1_5/eval/eval_task_aware_gate_iconqa.sh`

## Claim Guidance

Safe claim:

> A task-aware inference policy that routes IconQA to the target LoRA and source/general tasks to the base LLaVA checkpoint reaches `Avg=73.6825`, exceeding the reproduced LoRASculpt baseline by `+3.6288`.

Do not claim without qualification:

> A single static LoRASculpt checkpoint improves the average by `+3.6288`.

The current evidence does not support that static-checkpoint claim. Static post-hoc and training attempts either preserved target but gave only small source gains, or improved source by sacrificing IconQA.

## Mechanistic Interpretation

The experiments indicate that the main bottleneck is source/general forgetting under target LoRA activation:

- The target LoRA preserves IconQA well.
- The base model has much stronger source/general scores than the target LoRA on OKVQA, OCRVQA, GQA, and TextVQA.
- Small static LoRA delta injections were unable to recover enough source performance while preserving IconQA.
- Aggressive post-hoc delta pruning such as AdaDARE-gamma collapsed IconQA on this already target-tuned checkpoint.

The task-aware gate succeeds because it avoids forcing one parameter state to satisfy both target specialization and source retention.

## Negative Results Worth Reporting

These negative results justify the final routing policy and prevent overstating the contribution:

| Experiment line | Verdict | Evidence |
|---|---|---|
| MB-LDF post-hoc delta fusion | Target-safe but source lift too small. | `experiments/mbldf_plus1/results.csv` |
| SA-MIX mixed training | Source signal exists but target drops below gate. | `experiments/samix_plus1/results.csv` |
| SA-MIX delta fusion | Target preserved, source lift too small. | `experiments/samix_delta_fusion/results.csv` |
| TP-SA-MIX / PCGrad | Smoke can pass, but partial checkpoints collapse IconQA. | `experiments/tp_samix/results.csv` |
| OKVQA source-adapter fusion | OKVQA improves slightly, required GQA remains impossible. | `experiments/source_adapter_fusion/results.csv` |
| SPIDER | Blocked by DeepSpeed gradient lifecycle integration. | `experiments/spider/results.csv` |
| AdaDARE-gamma default | IconQA collapses to 38.47. | `experiments/adadare_gamma/results.csv` |

## Reproduction Commands

Full task-aware rerun:

```bash
cd /data/guoboyang/LoRa-Projects/LoRASculpt-repro/LoRASculpt
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/eval_task_aware_gate_iconqa.sh \
  --run-name taskaware_gamma090_target_base_source_YYYYMMDD_HHMM \
  --output-root /data/guoboyang/LoRa-Projects/LoRASculpt-repro/repro_results/task_aware_gamma
```

The committed result row was assembled from separately completed component evaluations to avoid rerunning the base source sweep:

```text
IconQA target LoRA: taskgate_gamma090_iconqa_20260508_0509
Base OKVQA/OCRVQA/TextVQA: taskgate_base_source3_20260508_0427
Base GQA: taskgate_base_gqa_20260508_0448
```

## Paper Writing Next Steps

1. Decide whether the paper contribution is allowed to be a task-aware inference policy.
2. If yes, write the main method as routing over LoRA activation conditioned on known task identity.
3. If no, continue static single-checkpoint research; current static best is only `Avg=70.09625`.
4. Add a limitations section: the best method requires task identity and stores/evaluates two parameter states.
5. Keep negative-result tables in appendix or ablation discussion.
