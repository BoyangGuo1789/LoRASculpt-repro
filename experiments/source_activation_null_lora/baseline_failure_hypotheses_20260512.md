# Baseline Failure Hypotheses for Internal LoRA Upgrades

Date: 2026-05-12

## Goal

Improve the LoRASculpt baseline by at least one average point while preserving a comparable inference setting: one model, one adapter, LoRA always active, no task gate, no checkpoint routing, and no evaluation-time LoRA disabling.

Exact reproduced baseline:

| Method | IconQA | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoRASculpt reproduced baseline | 86.26 | 52.71 | 54.70 | 56.34 | 51.64 | 53.8475 | 70.05375 |

Success threshold: `Avg >= 71.05375`.

## Rejected Low-Innovation Explanations

The previous LoRA-off / checkpoint-routing observation is treated as a diagnostic only. It shows that always-on target LoRA can hurt source/general tasks, but it is not a publishable method because the gain comes from choosing a different inference path rather than improving the baseline mechanism.

The next accepted method must be internal to the baseline training or LoRA component. Allowed directions include a scoring rule, regularizer, objective, adapter-internal constraint, pruning/training-time procedure, or activation/rank structure that still uses one adapter at inference.

## Evidence From Failed or Weak Directions

1. Direct source-answer mixed training preserved the source objective explicitly but hurt IconQA too much. This suggests the source and target answer losses compete strongly in the same low-rank adapter.
2. TP-SA-MIX had the same practical failure mode: source preservation through answer CE pulled capacity away from target specialization.
3. Static source-delta fusion only worked at tiny coefficients. Larger source deltas quickly damaged target performance, so naive parameter interpolation is too blunt.
4. RPB-LoRA improved IconQA to 86.64 but degraded every source/general task. This says rank-path support alone can sharpen target adaptation but does not solve always-on source perturbation.
5. PARS-style structural variants did not recover the required average gain and are now sealed as an explored branch.

## Current Baseline Limitation Hypothesis

LoRASculpt optimizes target specialization through a single always-on LoRA. Its pruning and regularization control parameter support, but they do not directly control where the learned LoRA function is active in activation space. As a result, target-beneficial LoRA updates also produce non-trivial deltas for source/general examples.

The failure is not simply "too much LoRA" or "wrong checkpoint at inference." The deeper issue is lack of activation-selective adapter behavior under the same inference path.

## Active Method Hypothesis: SAN-LoRA

Source-Activation Nulling LoRA trains the same always-on adapter to be quiet on source-style activations without using source answer CE. It uses COCO source-anchor examples only for activation regularization:

```text
L = L_target_ce + L_CMR
    + lambda_san * mean_m ||Delta_m(h_source)||_2^2
      / stopgrad(||Delta_m(h_target)||_2^2 + eps)
```

The intended effect is to preserve target LoRA energy on IconQA while suppressing unnecessary LoRA deltas on source/general activation regions.

## Decision Rule

Promote a SAN variant only if:

1. smoke training completes with finite losses and nonzero LoRA hook coverage,
2. full training writes adapter and non-LoRA trainables,
3. IconQA is at least the reproduced baseline gate (`>= 86.20` as a practical tolerance),
4. source/general evaluation lifts the final average above `71.05375`,
5. the gain is obtained with one checkpoint and one always-on adapter.

Reject or revise the direction if source/general metrics remain below baseline after two full SAN strengths, or if the only positive result requires task labels to choose another inference route.

