# Task-Aware LoRA Gate Experiments

This experiment tests a task-aware inference policy rather than a static single-checkpoint LoRASculpt model.

Policy:

- IconQA target: use the current best target LoRA checkpoint, `gamma090`.
- Source/general datasets: use the base LLaVA-1.5-7B checkpoint without the IconQA LoRA.

This is a valid diagnostic and an executable evaluation policy when task identity is known, but it should not be reported as a static LoRASculpt checkpoint result. It is useful because it quantifies how much of the remaining gap comes from source forgetting rather than target adaptation.

## Result

The policy reaches `Avg=73.6825`, exceeding the reproduced exact baseline `70.05375` by `+3.62875` and the requested plus-one target `71.05375` by `+2.62875`.

## Reproduction

A full rerun is available through:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/eval_task_aware_gate_iconqa.sh   --run-name taskaware_gamma090_target_base_source_YYYYMMDD_HHMM   --output-root /data/guoboyang/LoRa-Projects/LoRASculpt-repro/repro_results/task_aware_gamma
```

The committed row below was assembled from separately run target/source components to avoid rerunning the already completed base source sweep.
