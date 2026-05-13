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

## Current Best Full Result

| Method | Setting | IconQA | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg | Avg | Delta vs reproduced baseline |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TFR-BS no target KL | IconQA target, rank64 total, freeze32/residual32, OKVQA3k+COCO3k anchors, target_kl=0.0 | 86.78 | 59.04 | 59.85 | 56.20 | 52.93 | 57.0050 | 71.8925 | +1.83875 |

Previous target_kl=1.0 main configuration: IconQA=86.56, SourceAvg=57.2000,
Avg=71.8800. The no-KL ablation is only +0.0125 higher in full average, so both
variants should be treated as close until another seed or rank check confirms
the margin.

COCO-Caption probe for the target_kl=1.0 checkpoint: CIDEr=1.2138 (not a
COCO-target fine-tune).

This result keeps one checkpoint and one always-on PEFT adapter. It is not task
gating, checkpoint routing, or LoRA-off evaluation.


## Completed Ablations

| ID | Change | IconQA | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg | Avg | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A2a | Target teacher KL removed, OKVQA3k+COCO3k anchors unchanged | 86.78 | 59.04 | 59.85 | 56.20 | 52.93 | 57.0050 | 71.8925 | New best: +1.8388 vs reproduced baseline, +0.0125 vs target_kl=1.0. |
| A3a | COCO anchors 1500 instead of 3000, OKVQA anchors fixed at 3000 | 86.83 | 59.06 | 58.55 | 56.57 | 52.67 | 56.7125 | 71.7713 | Above baseline +1.7175; lower than COCO3000 main by 0.1088. |
| A4a | Freeze only 16 target ranks instead of 32, total rank64, target_kl=0.0 | 86.54 | 59.66 | 59.10 | 55.96 | 53.02 | 56.9350 | 71.7375 | Above baseline +1.6838; below A2a by 0.1550. |
| M4a | Total rank96 with freeze32/residual64, target_kl=0.0 | 86.81 | 59.71 | 59.05 | 56.19 | 52.52 | 56.8675 | 71.8388 | Above baseline +1.7850; below A2a by 0.0538. Capacity helps IconQA/OKVQA but not broad source retention. |
| A5a | Residual L2=1e-6 on rank64/freeze32, target_kl=0.0 | 86.80 | 59.03 | 59.20 | 56.07 | 53.06 | 56.8400 | 71.8200 | Above baseline +1.7663; below A2a by 0.0725. Regularization helps TextVQA but not enough to recover OKVQA/OCRVQA/GQA. |

A2a shows that the target-frozen block is doing the main target-preservation
work; explicit target-teacher KL is not necessary in this seed. The gain over
target_kl=1.0 is tiny, so it is useful as a component ablation but should not be
over-claimed as a separate method.

A3a shows that reducing COCO anchors improves IconQA and OKVQA, but hurts
OCRVQA/GQA/TextVQA enough that COCO3000 remains the better broad-retention
configuration.

A4a shows that relaxing the protected target subspace from 32 to 16 ranks is too
aggressive. It improves OKVQA to 59.66, but IconQA, OCRVQA, and GQA drop enough
that the full average trails A2a by 0.1550. This supports keeping a larger frozen
target block for the main method.

M4a shows that simply adding residual capacity is not enough. Rank96/freeze32
improves IconQA to 86.81 and OKVQA to 59.71, but OCRVQA, GQA, and TextVQA fall
enough that Avg=71.8388 trails A2a by 0.0538. This keeps rank64/freeze32 as the
main checkpoint and makes residual regularization or connector diagnostics more
valuable than another larger-rank sweep.

A5a shows that a small residual L2 penalty is also not enough to beat the main
configuration. It keeps IconQA high at 86.80 and improves TextVQA to 53.06, but
OKVQA, OCRVQA, and GQA trail A2a enough that Avg=71.8200 remains 0.0725 below
the current best. This rejects residual_l2=1e-6 as the default setting.

## Full-Suite Work Items

| ID | Baseline paper category | TFR-BS experiment | Status | Notes |
|---|---|---|---|---|
| M1 | Main IconQA target table | Reproduced baseline vs TFR-BS on IconQA + four source tasks | done | Full metrics already recorded. |
| M2 | COCO-Caption target table | COCO-Caption evaluation/probe for current IconQA-target TFR-BS | done | Probe CIDEr=1.2138; this is not a COCO-target fine-tune. |
| M3 | COCO-Caption downstream adaptation | Train a COCO-target TFR-BS analogue if a COCO LoRASculpt target checkpoint exists or can be trained | pending | Needed for strict Table-1 parity. |
| M4 | Rank scaling | Compare total rank/residual capacity variants | done | M4a rank96/freeze32 full eval done: Avg=71.8388, above reproduced baseline but below A2a by 0.0538. |
| D1 | Connector diagnostic | Check whether projector/non-LoRA trainables are necessary for TFR-BS | pending | Use `MM_PROJECTOR_LR=0` mainline vs controlled variants. |
| A1 | Component ablation | baseline target only, OKVQA-only residual, OKVQA+COCO residual | partial | Baseline and OKVQA-only are done; main TFR-BS is done. |
| A2 | Target preservation ablation | remove or weaken target KL | done | A2a target_kl=0 full eval is the current best: Avg=71.8925. |
| A3 | Source-anchor ablation | COCO samples 0/1500/3000 and possibly OKVQA samples 1500/3000 | partial | A3a COCO1500 full eval done: Avg=71.7713 (+1.7175 vs baseline), below COCO3000 main by 0.1088. |
| A4 | Frozen-rank ablation | freeze_rank 16/32 with total rank64 | done | A4a freeze_rank16 full eval done: Avg=71.7375, below A2a; freeze32 remains main. |
| A5 | Regularization ablation | residual L2 0/1e-6/1e-5 | done | A5a residual_l2=1e-6 full eval done: Avg=71.8200, above reproduced baseline but below A2a by 0.0725. |
| R1 | Epoch robustness | evaluate checkpoints or max-step variants across training progress | pending | Mirrors paper Fig. 3 qualitatively. |
| S1 | LoRA internal analysis | frozen/residual norm, delta energy, per-module residual movement | done | `s1_lora_delta_summary.json` confirms frozen-rank drift is 0.0 and residual ranks carry the learned movement. |

## Priority

1. `A2a` target_kl=0 is the current best full-average checkpoint, but the margin
   over target_kl=1.0 is only +0.0125.
2. `A3a` COCO1500 is complete; COCO3000 remains the better broad-retention
   configuration.
3. `A4a` indicates freeze_rank16 is not enough target protection; keep
   freeze_rank32 for the main method unless a larger total-rank variant restores
   IconQA/GQA.
4. `M4a` rank96/freeze32 and `A5a` residual_l2=1e-6 full evals are done.
   Neither improves overall Avg over A2a, so keep rank64/freeze32/tkl0/no-L2 as
   the main checkpoint. The next highest-value ablation is `D1` connector
   diagnostics rather than another capacity or weak-L2 sweep.
5. `S1` internal norm/delta analysis is now recorded; use it as method evidence
   before writing the method section.
6. Train/evaluate the strict COCO-target analogue only after confirming whether
   the baseline COCO LoRASculpt checkpoint is already available or must be
   reproduced from scratch.

## Logging Rules

Every completed item should add one row to `results.csv` and, when applicable,
copy compact metrics JSON into this directory. Large checkpoints, raw generated
answers, datasets, and logs stay outside git under `checkpoints/`,
`repro_results/`, and `logs/`.
