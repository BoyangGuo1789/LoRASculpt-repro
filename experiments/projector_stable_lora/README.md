# PSL: Projector-Stable LoRASculpt

Goal: start a non-gating, non-routing research line that stays inside the LoRASculpt baseline adaptation path.

## Baseline Limitation

The baseline trains target LoRA and non-LoRA `mm_projector` weights together, then applies both as a single static target adaptation. Prior runs show that strong target retention and source/general retention are in tension: source-aware LoRA masks or source-delta fusion either move source metrics too little or damage IconQA.

## Mechanism Hypothesis

Some of the source/general drop is caused by projector drift during target training. If the projector is kept stable and only the LoRA adapter carries target adaptation, the model may preserve more general visual-language alignment while keeping the IconQA target gain.

## Method

`PSL` keeps the target LoRA adapter unchanged and shrinks the saved `mm_projector` trainables toward the base LLaVA projector in a single checkpoint:

```text
language_model = base + target LoRA
mm_projector   = projector_base + alpha * (projector_target - projector_base)
```

This is not input gating, checkpoint routing, task-label routing, or LoRA deactivation. The checkpoint still has one target LoRA adapter that is active for every input. `alpha=0` is the projector-freezing diagnostic; `0<alpha<1` is the static version of a training-time projector-drift regularizer.

## Current Evidence

`alpha=0.0` failed the target gate with IconQA `55.54`, showing that fully restoring the base projector breaks the target LoRA/projector coupling and should not be promoted.

`alpha=0.95` and `alpha=0.98` both preserved IconQA, but the three-source gate showed that static projector shrink is too weak to reach the `baseline + 1` target:

| Variant | IconQA | OKVQA | OCRVQA | TextVQA | Required GQA for Avg >= 71.05375 | Decision |
|---|---:|---:|---:|---:|---:|---|
| alpha=0.95 | 86.30 | 53.27 | 56.75 | 52.35 | 60.86 | Reject |
| alpha=0.98 | 86.35 | 52.91 | 55.55 | 51.90 | 62.67 | Reject |

The required GQA is far above the known feasible range for this branch, so GQA was not run. PSL is useful as a diagnostic: the target LoRA and target-trained projector are tightly coupled, and small static projector interpolation cannot recover enough source/general capability. The next line should move from post-hoc projector interpolation to a training-time internal component change.

## Promotion Gate

Promote only if:

- IconQA remains close to the exact baseline target score (`>= 86.20` as the partial gate).
- Official five-task Avg reaches at least `71.05375`.
- The output checkpoint contains no `lora_input_gate_config.json`.
- Full eval logs confirm one static checkpoint path.

## Status

Sealed as a negative result on 2026-05-11. Do not continue tuning static projector alpha unless a later method provides a training-time projector constraint worth ablating.
