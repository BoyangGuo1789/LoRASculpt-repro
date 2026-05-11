# Projector-Anchored LoRA

## Baseline Limitation

LoRASculpt trains the target LoRA and the multimodal projector together, then applies both updates to every input. Prior evidence suggests the projector update is a major source/general forgetting path:

- `LoRA-IGP-Ponly` recovers source/general performance when the source path uses the base projector, while leaving target LoRA active.
- Static projector interpolation (`PSL alpha=0.95/0.98`) preserves IconQA but recovers too little source capability.
- Hard post-hoc projector reset (`PSL alpha=0`) fails IconQA, showing that target LoRA and target-trained projector are tightly co-adapted.

The missing test is whether training from the start with a fixed general projector lets LoRA absorb the target adaptation without creating projector drift.

## Method

Projector-Anchored LoRA keeps the baseline LoRASculpt trainer, CMR, pruning, rank, and inference path unchanged. The only training-procedure change is:

```text
mm_projector_lr = 0
```

The projector is still present in the same model, but its parameters remain anchored to the base LLaVA projector. Target specialization must be expressed through the LoRA adapter. This is not a task gate, checkpoint routing, or evaluation-time LoRA-off policy.

## Hypothesis

If projector drift is the dominant source/general damage mechanism, freezing the projector during target training should preserve more source/general capability than the baseline while still allowing LoRA to recover IconQA.

## Gate

1. Smoke must complete with finite loss and write adapter/non-LoRA files.
2. Full training is promoted only if smoke preserves normal LoRASculpt behavior: one prune event and no OOM/NaN.
3. IconQA gate must be `>= 86.20`.
4. Source/general eval only runs after IconQA gate passes.
5. Success requires `Avg >= 71.05375` with one static checkpoint and always-on LoRA.

## Smoke Result

`pal_mm0_smoke20_20260512_053343` passed:

- command used `MM_PROJECTOR_LR=0`, `MAX_STEPS=20`, and `STEP_THRESHOLD=1`
- checkpoint: `/data/guoboyang/LoRa-Projects/LoRASculpt-repro/checkpoints/llava-v1.5-7b-lorasculpt-pal-mm0-r32-smoke-20260512_053343`
- log: `/data/guoboyang/LoRa-Projects/LoRASculpt-repro/logs/pal_mm0_smoke20_20260512_053343.log`
- `train_loss=0.5181412443518638`
- `adapter_model.bin` and `non_lora_trainables.bin` were written

Next step: full train with the same projector-anchored setting, then IconQA gate before source/general evaluation.
