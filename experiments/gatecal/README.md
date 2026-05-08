# LoRASculpt GateCal

## Baseline Limitation

The baseline trains one target LoRA and then keeps a fixed top-weight mask. The resulting adapter is always active at inference, so source/general tasks see the same target-specialized residual that helps IconQA. Previous static fixes either preserved IconQA with almost no source recovery or moved source enough to collapse IconQA.

## Method

`LoRASculptGateCal` is a module-wise static LoRA residual calibration step. Starting from the trained baseline IconQA LoRA, it freezes the base model and all LoRA matrices, then learns one scalar gate for each LoRA B module on a mixed IconQA + COCO calibration set. Target loss keeps useful IconQA residuals active; source-anchor loss suppresses modules whose target residual hurts general capability.

After calibration, the learned gates are baked into the LoRA B weights. The saved artifact is a normal single static LoRA checkpoint and does not need task labels or checkpoint routing at inference.

`GateCal-TR` is the trust-region post-hoc variant used after full GateCal proved too aggressive. It reuses the learned gate direction but blends it toward the original baseline adapter:

```text
LoRA_B' = LoRA_B * (1 - alpha * (1 - learned_gate))
```

This tests whether the calibration direction can recover source capability without collapsing IconQA.

## Planned Gates

- Smoke first: `SMOKE=1 MAX_STEPS=20`.
- Stable runtime: `USE_DEEPSPEED=0`, `GRADIENT_CHECKPOINTING=False`, `PER_DEVICE_TRAIN_BATCH_SIZE=1`.
  DeepSpeed Zero2 cannot partition only scalar gate parameters cleanly, and DDP
  with reentrant checkpointing marks the same gate ready twice.
- Full candidates:
  - `coco1500`, 200 gate steps, LR `5e-2`.
  - `coco3000`, 200 gate steps, LR `5e-2`.
- Promote only if IconQA stays at least `86.20` and partial source tasks imply required GQA at or below `57.0` for `Avg >= 71.05375`.

## Result Ledger

See `results.csv`.

## 2026-05-08 GateCal-TR Outcome

Full 200-step GateCal collapsed IconQA to about 55.6, so it is not a viable static adapter. GateCal-TR alpha `0.10` and `0.20` preserved IconQA:

- `coco1500-a010`: IconQA `86.27`.
- `coco1500-a020`: IconQA `86.26`, OKVQA `52.61`, OCRVQA `54.65`, TextVQA `51.67`; required GQA for `Avg >= 71.05375` is `64.46`.
- `coco3000-a010`: IconQA `86.26`.
- `coco3000-a020`: IconQA `86.30`, OKVQA `52.62`, OCRVQA `54.75`, TextVQA `51.72`; required GQA for `Avg >= 71.05375` is `64.14`.

Conclusion: GateCal-TR is target-safe but too weak. It should not be promoted to full GQA evaluation unless a stronger source-side component is added.
