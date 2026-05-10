# LoRA-IG: Input-Gated LoRA

Goal: address a baseline limitation inside the LoRA path rather than by external checkpoint routing.

## Baseline Limitation

The LoRASculpt target adapter is applied uniformly to every input. This is good for IconQA, but the same target delta is also forced onto source/general VQA prompts where it can overwrite useful base-model behavior.

## Mechanism Hypothesis

The model should not treat the target LoRA delta as a constant global offset. A lightweight input-conditioned gate can keep the LoRA active for target-style inputs while attenuating it for source-style inputs, preserving the same base model and a single adapter checkpoint.

## Method

`LoRA-IG` stores a compact gate config inside a normal PEFT LoRA checkpoint. During loading, LoRA layers remain unmerged. Each PEFT LoRA Linear computes the usual delta, then multiplies that delta by a continuous gate derived from the input prompt:

`y = W x + g(input) * B A x`

The first implementation trains a linear bag-of-text-features gate from IconQA train prompts versus OKVQA train prompts. It does not use eval labels, external datasets, new pretrained models, task labels at inference, or checkpoint switching.

## LoRA-IGP Variant

`LoRA-IGP` extends the same internal gate to the saved projector adaptation:

`projector(x) = projector_base(x) + g(input) * (projector_target(x) - projector_base(x))`

This tests the evidence from `LoRA-IG`: source/general recovery stayed weak even when LoRA gate was near zero, implying that non-LoRA projector adaptation may also be over-applied to source inputs. It is still a single model path with an input-conditioned adaptation module, not external checkpoint routing.

## Gate

Promote only if:

- IconQA remains comparable to the static target LoRA baseline.
- Source/general tasks recover enough for Avg >= 71.05375.
- The saved checkpoint contains exactly one LoRA adapter plus the internal gate config.
