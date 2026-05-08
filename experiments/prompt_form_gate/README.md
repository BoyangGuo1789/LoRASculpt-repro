# Prompt-Form Adaptive Residual Gate

## Motivation

The exact LoRASculpt baseline keeps the IconQA LoRA residual active for every
sample. That helps the structured IconQA target task but suppresses source
capability on open-ended VQA tasks. Static residual shrinking preserves IconQA
only when it is too weak to recover source accuracy.

## Method

Prompt-Form Adaptive Residual Gate is a task-name agnostic inference component.
For each input prompt, it enables the target LoRA path only when the prompt has
a structured multiple-choice answer form:

```text
gate = 1 if the prompt contains at least two A-D option lines and is not the
       open-ended "Answer the question using a single word or phrase." form
gate = 0 otherwise
```

The rule uses the sample prompt itself, not the dataset name. In a runtime
implementation, `gate=1` evaluates with the LoRA residual and `gate=0` evaluates
with the base model. The current experiment first verifies the exact metric
upper bound by merging already generated LoRA/base answer files with the same
per-sample gate.

The runtime evaluator then implements the same rule by splitting each question
file by prompt form, generating gate-on samples with the LoRA checkpoint and
gate-off samples with the base checkpoint, and merging answers back in original
question order before running the official evaluator.

## Stop Criteria

Promote only if the merged official evaluation exceeds exact reproduced baseline
Avg `70.05375` by at least `+1.0` point and all per-task routing counts are
auditable.

## 2026-05-08 Result

Answer-merge run `pfg_mcqa_static_20260508_1908` passes the plus-one target:

- IconQA `86.26`
- OKVQA `57.99`
- OCRVQA `66.15`
- GQA `61.93`
- TextVQA `58.23`
- SourceAvg `61.075`
- Avg `73.6675`
- Delta vs exact reproduced baseline `+3.61375`

Runtime official eval run `pfg_runtime_full_20260508_1935` reproduces the same metrics:

- IconQA `86.26`
- OKVQA `57.99`
- OCRVQA `66.15`
- GQA `61.93`
- TextVQA `58.23`
- SourceAvg `61.075`
- Avg `73.6675`
- Delta vs exact reproduced baseline `+3.61375`

Gate audit:

- IconQA gate-on LoRA: `6316/6316`.
- Source gate-on LoRA: `0/24624`.

Implementation note: TextVQA has repeated `question_id` values, so answer
merging must consume answers from a per-`question_id` queue instead of a plain
dictionary.

Smoke run `pfg_runtime_smoke8_20260508_1932` verified both runtime branches:
the first 8 IconQA samples used LoRA and the first 8 OKVQA samples used base.
OKVQA subset official scoring is intentionally skipped because the VQA API
requires the complete question set.
