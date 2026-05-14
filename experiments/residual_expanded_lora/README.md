# Residual-Expanded LoRA

## Baseline Limitation

Recent static LoRA attempts show a consistent conflict: changes that preserve IconQA tend to move source/general scores only slightly, while stronger source-preserving changes damage target performance. RPB showed that preserving all rank paths can improve IconQA but still worsens source/general scores. PAL showed that freezing the projector removes only a small part of the damage.

The working hypothesis is that the rank-32 LoRASculpt adapter is forced to store target-specific and source/general corrections in the same compressed subspace. SVD-compressed or globally interpolated source deltas are too blunt: either they are tiny and harmless, or they interfere with target-critical directions.

## Method

Residual-Expanded LoRA keeps one static PEFT adapter and one always-on inference path. It preserves the target LoRA block exactly, then appends a scaled source/general residual block as extra ranks:

    Delta_out = target_gamma * Delta_target + source_lambda * Delta_source

With rank32 target and rank32 source adapters, the first experiment writes a rank64 adapter with `lora_alpha=128`, preserving the original LoRA scaling. There is no task gate, checkpoint routing, or evaluation-time LoRA-off behavior.

The intended mechanism is not to choose a source checkpoint at inference, but to add capacity inside the LoRA factorization so target and source corrections do not need to be compressed into the same rank-32 basis.

## First Gate

Use the target-safe reproduced IconQA adapter as the target block and the OKVQA source adapter as the residual block. Start with a conservative source coefficient and evaluate IconQA first.

Promotion requires:

- checkpoint loads as one PEFT adapter;
- IconQA >= 86.20;
- source/general partial scores indicate a plausible path to `Avg >= 71.05375`;
- full eval only after IconQA passes.

## Command Template

    python scripts/v1_5/tools/rank_expanded_residual_lora.py \
      --target-checkpoint <target> \
      --source-checkpoint <source> \
      --output-checkpoint <output> \
      --scope qv --layer-band mid_late \
      --source-lambda 0.02 \
      --overwrite

Then run the normal official Issue2 evaluation wrapper with one checkpoint and always-on LoRA.

## 2026-05-12 Result

REL validated the capacity hypothesis only partially. Expanding the adapter to
rank 64 and appending an OKVQA residual can lift OKVQA while preserving IconQA:

| Run | Scope | Lambda | IconQA | OKVQA | OCRVQA | GQA | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `rel_qv_midlate_l002` | q/v mid-late | 0.02 | 86.29 | 52.75 | 54.75 | - | reject |
| `rel_qv_midlate_l005` | q/v mid-late | 0.05 | 86.27 | 52.85 | 54.70 | - | reject |
| `rel_qv_midlate_l010` | q/v mid-late | 0.10 | 86.29 | 53.06 | - | - | reject |
| `rel_all_midlate_l010` | all projections mid-late | 0.10 | 86.23 | 54.13 | - | - | parked |
| `rel_all_midlate_l020` | all projections mid-late | 0.20 | 86.24 | 55.32 | 54.40 | 55.90 | reject |

The best REL candidate improved OKVQA by `+2.61`, but OCRVQA and GQA dropped
below baseline. Given IconQA `86.24`, OKVQA `55.32`, OCRVQA `54.40`, and GQA
`55.90`, TextVQA would need `57.85` to reach `Avg >= 71.05375`. That is not a
plausible continuation, so TextVQA was stopped and REL is sealed as a useful
diagnostic rather than the final direction.

Mechanism takeaway: static rank expansion can carry an OKVQA-specific residual
without immediately breaking IconQA, but post-hoc residual insertion does not
learn a balanced source/general correction. The next direction should train the
residual branch under a target-preservation constraint instead of copying a
source adapter into the target adapter.
