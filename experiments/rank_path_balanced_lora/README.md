# Rank-Path Balanced LoRASculpt

## Baseline Limitation

The LoRASculpt baseline prunes LoRA parameters with a global element top-k rule. A LoRA update is a sum of rank-1 paths, but global element pruning can leave some A rows or B columns nearly inactive. That creates a target-capacity bottleneck after pruning and can make the single always-on adapter brittle across target and source/general tasks.

## Method

`RPB-LoRA` keeps the LoRASculpt training and inference contract unchanged: one checkpoint, one LoRA path, no task-aware routing, no checkpoint selection, and no LoRA-off evaluation.

The only change is the mask selection rule when `MIGDIS_SELECTION_MODE=rpb`:

1. Compute the existing MIG-DIS/LoRASculpt score for each trainable LoRA A/B parameter.
2. Reserve a configurable fraction of the keep budget for rank-path balancing.
3. For `lora_A`, keep the top scored entries in every rank row.
4. For `lora_B`, keep the top scored entries in every rank column.
5. Fill the remaining budget with the global top scored entries.
6. Freeze the resulting mask for the rest of training, as in LoRASculpt.

The intended mechanism is to preserve every low-rank path's ability to express target adaptation while still using the baseline score to prefer important parameters.

## Default Smoke Command

```bash
DEVICE=localhost:0,1,2,3 MASTER_PORT=29680 \
RUN_NAME=llava-v1.5-7b-lorasculpt-rpb-q060-r32 \
SMOKE=1 SMOKE_MAX_STEPS=20 \
MIGDIS_ENABLE=True MIGDIS_SELECTION_MODE=rpb \
MIGDIS_RPB_QUOTA_FRAC=0.60 MIGDIS_RPB_MIN_PER_RANK=1 \
bash scripts/v1_5/train/ours-train-migdis-official-issue2-iconqa.sh
```

## Promotion Gate

Promote to full training only if smoke shows:

- one prune event at the expected step;
- active ratio remains close to `AB_PRESERVE_RATIO`;
- rank nonzero fraction is `1.0` for A and B on tracked LoRA modules;
- no NaN/OOM;
- checkpoint files are written.

Then evaluate IconQA first. Full source/general eval only makes sense if IconQA is at least comparable to the exact baseline gate.

## Current Status

`rpb_q060_smoke20_20260511_223140`, `rpb_q040_smoke20_20260511_230217`, and `rpb_q080_smoke20_20260511_231434` passed smoke:

- one prune at `global_step=1`;
- active ratio A/B: `0.100005`;
- A/B rank nonzero coverage: `1.0` across 224 modules;
- final 20-step train loss stayed finite (`0.5084` to `0.5145`);
- checkpoint and `migdis_mask_stats.json` written.

`rpb_q060_full_eval` is rejected as a source-forgetting failure:

| Metric | Score |
|---|---:|
| IconQA | 86.64 |
| OKVQA | 52.12 |
| OCRVQA | 51.65 |
| GQA | 55.82 |
| TextVQA | 49.44 |
| SourceAvg | 52.2575 |
| Avg | 69.44875 |

Interpretation: RPB preserved and slightly improved target IconQA, but it made source/general tasks worse than the exact reproduced baseline. This falsifies the hypothesis that rank-path coverage alone fixes LoRASculpt's cross-task forgetting. The next method should target a deeper limitation than static LoRA support geometry.
