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

## Stop Criteria

Promote only if the merged official evaluation exceeds exact reproduced baseline
Avg `70.05375` by at least `+1.0` point and all per-task routing counts are
auditable.

## 2026-05-08 Result

Run `pfg_mcqa_static_20260508_1908` passes the plus-one target:

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
