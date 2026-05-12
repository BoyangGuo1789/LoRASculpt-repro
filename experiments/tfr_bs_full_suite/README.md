# TFR-BS Full Experiment Suite

This directory tracks the comprehensive experiment stage for Target-Frozen
Residual LoRA with Balanced Source Anchors (TFR-BS). The goal is to cover the
same experiment categories used by the LoRASculpt baseline paper, then add
method-specific ablations that support the TFR-BS mechanism.

## Baseline Paper Experiment Matrix

The LoRASculpt paper evaluates downstream knowledge acquisition on two target
tasks and source/general retention on four upstream tasks:

- target tasks: `IconQA`, `COCO-Caption`;
- source tasks: `OKVQA`, `OCRVQA`, `GQA`, `TextVQA`;
- aggregate metrics: `SourceAvg`, `Target`, `Avg=(SourceAvg+Target)/2`;
- LoRA ranks: `16`, `32`, `64`;
- diagnostic categories: connector adaptation, component ablation,
  hyperparameter ablation, epoch robustness, and LoRA sparsity analysis.

The paper reference values used for planning are extracted from the CVPR 2025
paper text and the local reproduction helper `scripts/repro_compare_with_paper.py`.
For this project, the exact reproduced baseline for the active IconQA setting is:

| Method | IconQA | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoRASculpt reproduced baseline | 86.26 | 52.71 | 54.70 | 56.34 | 51.64 | 53.8475 | 70.05375 |

## Current Main Result

| Method | Setting | IconQA | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg | Avg | Delta vs reproduced baseline |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TFR-BS | IconQA target, rank64 total, freeze32/residual32, OKVQA3k+COCO3k anchors | 86.56 | 58.64 | 60.25 | 56.78 | 53.13 | 57.2000 | 71.8800 | +1.82625 |

COCO-Caption probe for the same checkpoint: CIDEr=1.2138 (not a COCO-target fine-tune).

This result keeps one checkpoint and one always-on PEFT adapter. It is not task
gating, checkpoint routing, or LoRA-off evaluation.


## Completed Ablations

| ID | Change | IconQA | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg | Avg | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A3a | COCO anchors 1500 instead of 3000, OKVQA anchors fixed at 3000 | 86.83 | 59.06 | 58.55 | 56.57 | 52.67 | 56.7125 | 71.7713 | Above baseline +1.7175; lower than COCO3000 main by 0.1088. |

A3a shows that reducing COCO anchors improves IconQA and OKVQA, but hurts
OCRVQA/GQA/TextVQA enough that the COCO3000 main configuration remains the best
current full-average checkpoint.

## Full-Suite Work Items

| ID | Baseline paper category | TFR-BS experiment | Status | Notes |
|---|---|---|---|---|
| M1 | Main IconQA target table | Reproduced baseline vs TFR-BS on IconQA + four source tasks | done | Full metrics already recorded. |
| M2 | COCO-Caption target table | COCO-Caption evaluation/probe for current IconQA-target TFR-BS | done | Probe CIDEr=1.2138; this is not a COCO-target fine-tune. |
| M3 | COCO-Caption downstream adaptation | Train a COCO-target TFR-BS analogue if a COCO LoRASculpt target checkpoint exists or can be trained | pending | Needed for strict Table-1 parity. |
| M4 | Rank scaling | Compare total rank/residual capacity variants | pending | Proposed: r48 freeze32, r64 freeze32, r96 freeze32. |
| D1 | Connector diagnostic | Check whether projector/non-LoRA trainables are necessary for TFR-BS | pending | Use `MM_PROJECTOR_LR=0` mainline vs controlled variants. |
| A1 | Component ablation | baseline target only, OKVQA-only residual, OKVQA+COCO residual | partial | Baseline and OKVQA-only are done; main TFR-BS is done. |
| A2 | Target preservation ablation | remove or weaken target KL | pending | Tests whether target-teacher KL is needed. |
| A3 | Source-anchor ablation | COCO samples 0/1500/3000 and possibly OKVQA samples 1500/3000 | partial | A3a COCO1500 full eval done: Avg=71.7713 (+1.7175 vs baseline), below COCO3000 main by 0.1088. |
| A4 | Frozen-rank ablation | freeze_rank 16/32 with total rank64 | pending | Tests target block protection strength. |
| A5 | Regularization ablation | residual L2 0/1e-6/1e-5 | pending | Tests whether residual drift control helps. |
| R1 | Epoch robustness | evaluate checkpoints or max-step variants across training progress | pending | Mirrors paper Fig. 3 qualitatively. |
| S1 | LoRA internal analysis | frozen/residual norm, delta energy, per-module residual movement | pending | Replaces LoRASculpt sparsity theorem plots with TFR-BS mechanism evidence. |

## Priority

1. `M2` COCO-Caption probe is complete for the current successful checkpoint.
2. `A3a` COCO1500 is complete; COCO3000 remains the main configuration.
3. Run `A2`: target KL is the cleanest target-preservation component ablation.
4. Run `A4` or `A5` only if the first two ablations leave the mechanism unclear.
5. Train/evaluate the strict COCO-target analogue only after confirming whether
   the baseline COCO LoRASculpt checkpoint is already available or must be
   reproduced from scratch.

## Logging Rules

Every completed item should add one row to `results.csv` and, when applicable,
copy compact metrics JSON into this directory. Large checkpoints, raw generated
answers, datasets, and logs stay outside git under `checkpoints/`,
`repro_results/`, and `logs/`.
