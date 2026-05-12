# Target-Frozen Residual LoRA

TFR-LoRA is the next internal baseline-upgrade direction after static residual
expansion. It keeps the reproduced IconQA LoRASculpt adapter as a frozen target
block and appends new residual ranks inside the same LoRA matrices.

At inference there is still one PEFT adapter and LoRA is always active:

```text
Delta(x) = B_t A_t x + B_r A_r x
```

The first block `(A_t, B_t)` is copied from the target LoRASculpt baseline and
its gradients are masked to zero. The residual block `(A_r, B_r)` starts as a
zero-function branch and is trained on an IconQA + OKVQA mix with target-teacher
KL. The intended mechanism is to prevent source/general corrections from
overwriting the baseline target subspace while still allowing a learned LoRA
component to recover source capability.

This is not task gating, checkpoint routing, or evaluation-time LoRA disabling.
The comparison target remains the exact reproduced LoRASculpt baseline:

| Method | IconQA | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoRASculpt reproduced baseline | 86.26 | 52.71 | 54.70 | 56.34 | 51.64 | 53.8475 | 70.05375 |

Success threshold: `Avg >= 71.05375`.

## Gate

1. Build a rank-64 initialization whose first 32 ranks exactly reproduce the
   target adapter and whose residual branch has zero functional contribution.
2. Smoke train for 20 steps and verify finite loss plus TFR gradient-mask logs.
3. Run IconQA gate first. Continue only if IconQA is at least 86.20.
4. Run OKVQA gate. Continue to OCRVQA/TextVQA/GQA only if the source lift is
   plausibly large enough for the +1 average target.

## 2026-05-12 Smoke

The fixed smoke run `tfr_okvqa_s3000_sw1_tkl1_smoke20fastfix_20260512_103753`
passed the implementation gate:

- rank-64 target initialization loaded;
- target non-LoRA trainables loaded from the reproduced baseline checkpoint;
- 448 LoRA tensors received gradient masks;
- frozen target-block values: `79,953,920`;
- residual trainable values: `79,953,920`;
- train loss: `0.50484`;
- post-train target-block max absolute diff: `0.0`;
- post-train residual-block max absolute diff: `5.455e-4`.

The first smoke attempt showed that gradient masks alone were insufficient under
the optimizer path, so TFR now restores frozen slices before each forward pass
and again before final adapter saving. The reference slices live on the same
device as the LoRA parameters to avoid per-step CPU-to-GPU copies.

## 2026-05-12 Full OKVQA-Residual Result

Run `tfr_lora_s3000_sw1_tkl1_full_20260512_104659` trained for one epoch on
IconQA plus 3,000 OKVQA source examples with source weight `1.0`, target KL
`1.0`, and frozen first 32 ranks.

Post-training adapter verification passed:

- root adapter keys: `448`;
- teacher keys in root adapter: `0`;
- frozen target-block values checked: `79,953,920`;
- residual-block values checked: `79,953,920`;
- target-block max absolute diff vs init: `0.0`;
- residual-block max absolute diff vs init: `7.93e-3`;
- train loss: `0.23130`.

| Method | IconQA | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg | Avg | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TFR-LoRA OKVQA residual | 86.42 | 57.77 | 54.20 | 55.61 | 52.38 | 54.9900 | 70.7050 | +0.65125 |

Verdict: useful but not sufficient. TFR validates the core mechanism because it
raises IconQA by `+0.16` and OKVQA by `+5.06` under one always-on adapter, but
it loses `0.50` on OCRVQA and `0.73` on GQA. The next variant should keep the
target-frozen residual structure while training the residual branch on a broader
source mixture or adding a source-balance constraint so that the OKVQA gain does
not come at the expense of other source/general datasets.

## 2026-05-12 Balanced-Source Smoke

TFR-BS keeps the same target-frozen residual adapter but extends the residual
training mix from IconQA + OKVQA to IconQA + OKVQA + COCO caption source
anchors. The hypothesis is that COCO anchors may recover broad visual-language
coverage without changing inference: one model, one adapter, LoRA always on.

Smoke run `tfr_bs_ok3k_coco3k_smoke20_20260512_124658` used:

- IconQA: `10,000`;
- OKVQA train samples: `3,000`;
- COCO caption samples: `3,000`;
- rank/alpha/freeze: `64/128/32`;
- source weight: `1.0`;
- target KL: `1.0`;
- train loss: `0.98270`.

Adapter verification passed:

- root adapter keys: `448`;
- teacher keys in root adapter: `0`;
- target-block max absolute diff vs init: `0.0`;
- residual-block max absolute diff vs init: `5.493e-4`.

This smoke only verifies implementation and target-rank freezing. It does not
yet establish metric value; the next gate is a full TFR-BS run followed by
IconQA and OKVQA evaluation.

## 2026-05-12 Full Balanced-Source Result

Run `tfr_bs_ok3k_coco3k_full_20260512_125734` trained the same target-frozen
residual adapter for one epoch on IconQA + 3,000 OKVQA + 3,000 COCO caption
anchors. The inference setting remains unchanged: one checkpoint, one PEFT
adapter, LoRA always active, no task gate and no checkpoint routing.

Post-training adapter verification passed:

- root adapter keys: `448`;
- teacher keys in root adapter: `0`;
- frozen target-block values checked: `79,953,920`;
- residual-block values checked: `79,953,920`;
- target-block max absolute diff vs init: `0.0`;
- residual-block max absolute diff vs init: `1.465e-2`;
- train loss: `0.49589`.

| Method | IconQA | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg | Avg | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRASculpt reproduced baseline | 86.26 | 52.71 | 54.70 | 56.34 | 51.64 | 53.8475 | 70.05375 | 0.00000 |
| TFR-BS OKVQA+COCO residual | 86.56 | 58.64 | 60.25 | 56.78 | 53.13 | 57.2000 | 71.8800 | +1.82625 |

Verdict: success. TFR-BS exceeds the reproduced LoRASculpt baseline by
`+1.82625` average points and clears the required `baseline +1` threshold. The
important mechanism evidence is that the frozen target rank block keeps IconQA
above baseline while the residual ranks recover source/general capability across
all four evaluated source tasks. The COCO caption anchors are useful here not as
a new task-specific route, but as broad source activation coverage for the same
always-on residual LoRA branch.
