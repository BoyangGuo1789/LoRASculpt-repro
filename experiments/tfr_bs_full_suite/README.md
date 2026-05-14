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

## Strict COCO-Target Baseline

The strict COCO-Caption LoRASculpt baseline is now reproduced from the official
issue2 10k COCO-Caption JSON with rank32 LoRA and the same always-on inference
setting. `SourceAvg` is averaged over the four source accuracies. The CSV leaves
`Avg` and `delta_vs_reproduced_baseline` blank for this row because COCO CIDEr is
not on the same scale as IconQA accuracy.

| Method | COCO CIDEr | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg |
|---|---:|---:|---:|---:|---:|---:|
| LoRASculpt COCO-target reproduced baseline | 1.1667 | 5.96 | 58.35 | 58.15 | 26.27 | 37.1825 |
| TFR-BS COCO-target + OKVQA3k anchors | 1.0784 | 10.52 | 59.30 | 54.71 | 33.22 | 39.4375 |

This establishes the M3 comparison for strict COCO-target adaptation. The
matched TFR-BS analogue improves SourceAvg by +2.2550 over the reproduced
LoRASculpt COCO baseline, mainly through OKVQA (+4.56), TextVQA (+6.95), and
OCRVQA (+0.95), but it lowers COCO CIDEr by -0.0883 and GQA by -3.44. This is a
useful stress-test result rather than a new best target-task setting: balanced
source anchors recover part of the source loss under always-on inference, while
COCO caption target quality remains better with the original LoRASculpt
fine-tune.



## Table 1 Rank-16/32 Completion

The baseline-style Table 1 rank sweep for `XLora` is now complete for total
ranks 16 and 32. The IconQA target rows use `Avg=(SourceAvg+IconQA)/2`. The COCO
rows report CIDEr in the ledger and CIDEr x100 in the paper table; the table
average follows the LoRASculpt formatting convention, while `results.csv` keeps
COCO `Avg` blank because CIDEr and VQA accuracy are on different scales.

| Target | Rank | Frozen target rank | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg | Target | Table Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IconQA | 32 | 16 | 59.84 | 59.20 | 55.65 | 53.10 | 56.9475 | 86.56 | 71.7538 |
| IconQA | 16 | 8 | 59.14 | 58.75 | 55.88 | 51.82 | 56.3975 | 86.68 | 71.5388 |
| COCO-Caption | 32 | 16 | 57.44 | 60.55 | 54.07 | 50.23 | 55.5725 | 109.35 | 82.4613 |
| COCO-Caption | 16 | 8 | 56.94 | 60.05 | 54.11 | 49.87 | 55.2425 | 109.99 | 82.6163 |

The compact run summary is saved in
`table1_rank32_rank16_xlora_metrics.json`. All four evaluations used GPUs
`0,1,3,4,5,6,7`, avoiding the throttled GPU 2.

## Completed Ablations

| ID | Change | IconQA | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg | Avg | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A2a | Target teacher KL removed, OKVQA3k+COCO3k anchors unchanged | 86.78 | 59.04 | 59.85 | 56.20 | 52.93 | 57.0050 | 71.8925 | New best: +1.8388 vs reproduced baseline, +0.0125 vs target_kl=1.0. |
| A3a | COCO anchors 1500 instead of 3000, OKVQA anchors fixed at 3000 | 86.83 | 59.06 | 58.55 | 56.57 | 52.67 | 56.7125 | 71.7713 | Above baseline +1.7175; lower than COCO3000 main by 0.1088. |
| A4a | Freeze only 16 target ranks instead of 32, total rank64, target_kl=0.0 | 86.54 | 59.66 | 59.10 | 55.96 | 53.02 | 56.9350 | 71.7375 | Above baseline +1.6838; below A2a by 0.1550. |
| M4a | Total rank96 with freeze32/residual64, target_kl=0.0 | 86.81 | 59.71 | 59.05 | 56.19 | 52.52 | 56.8675 | 71.8388 | Above baseline +1.7850; below A2a by 0.0538. Capacity helps IconQA/OKVQA but not broad source retention. |
| A5a | Residual L2=1e-6 on rank64/freeze32, target_kl=0.0 | 86.80 | 59.03 | 59.20 | 56.07 | 53.06 | 56.8400 | 71.8200 | Above baseline +1.7663; below A2a by 0.0725. Regularization helps TextVQA but not enough to recover OKVQA/OCRVQA/GQA. |
| D1a | Do not load target non-lora/projector state, rank64/freeze32, target_kl=0.0 | 80.11 | 61.36 | 66.25 | 58.94 | 55.91 | 60.6150 | 70.3625 | Diagnostic only: source retention rises sharply, but IconQA drops by 6.67 vs A2a; not a comparable main result. |

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

D1a shows that the target non-lora/projector state is a necessary connector for
the comparable always-on setting. Removing it raises SourceAvg to 60.6150, but
IconQA collapses to 80.11, which is 6.67 points below A2a. This is useful
evidence for the baseline limitation and target/source tradeoff, but it is not
a publishable method result because the target task is no longer preserved.

## Full-Suite Work Items

| ID | Baseline paper category | TFR-BS experiment | Status | Notes |
|---|---|---|---|---|
| M1 | Main IconQA target table | Reproduced baseline vs TFR-BS on IconQA + four source tasks | done | Full metrics already recorded. |
| M2 | COCO-Caption target table | COCO-Caption evaluation/probe for current IconQA-target TFR-BS | done | Probe CIDEr=1.2138; this is not a COCO-target fine-tune. |
| M3 | COCO-Caption downstream adaptation | Strict COCO-target LoRASculpt baseline, then matched COCO-target TFR-BS analogue | done | Baseline: CIDEr=1.1667, SourceAvg=37.1825. TFR-BS analogue: CIDEr=1.0784, SourceAvg=39.4375. |
| M4 | Rank scaling | Compare total rank/residual capacity variants | done | M4a rank96/freeze32 full eval done: Avg=71.8388, above reproduced baseline but below A2a by 0.0538. |
| T1 | Main Table 1 rank sweep | XLora total ranks 16 and 32 for IconQA and COCO-Caption targets | done | Rank32: IconQA Avg=71.7538, COCO table Avg=82.4613. Rank16: IconQA Avg=71.5388, COCO table Avg=82.6163. |
| D1 | Connector diagnostic | Check whether projector/non-LoRA trainables are necessary for TFR-BS | done | D1a LOAD_NON_LORA=False full eval done: SourceAvg=60.6150 but IconQA=80.11, confirming target connector state is necessary. |
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
4. `M4a` rank96/freeze32, `A5a` residual_l2=1e-6, and `D1a` no-non-lora
   connector diagnostic full evals are done. None improves the comparable
   target-preserving average over A2a, so keep rank64/freeze32/tkl0/no-L2 with
   loaded target non-lora/projector state as the main checkpoint.
5. `S1` internal norm/delta analysis is now recorded; use it as method evidence
   before writing the method section.
6. The strict COCO-target comparison is now complete. TFR-BS improves SourceAvg
   from 37.1825 to 39.4375 but lowers CIDEr from 1.1667 to 1.0784, so the paper
   should present it as a source-retention stress test, not as a target-metric
   win.
7. The Table 1 rank16/rank32 sweep is complete for the paper-facing `XLora`
   name. Rank32 gives the stronger IconQA table average, while rank16 gives a
   slightly stronger COCO table average in this run.

## Logging Rules

Every completed item should add one row to `results.csv` and, when applicable,
copy compact metrics JSON into this directory. Large checkpoints, raw generated
answers, datasets, and logs stay outside git under `checkpoints/`,
`repro_results/`, and `logs/`.
