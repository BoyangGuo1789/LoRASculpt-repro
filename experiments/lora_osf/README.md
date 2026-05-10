# LoRA-OSF: Orthogonal Source-Delta Fusion

Goal: test a real LoRA-internal improvement over the reproduced LoRASculpt baseline, with a target of at least +1.0 average point over the exact static baseline.

## Baseline Limitation

The reproduced LoRASculpt target adapter preserves IconQA but suppresses source/general VQA ability. Prior small source-adapter delta fusion was target-safe, but the source signal was too weak because the injected source update had to stay at tiny global coefficients. Mixed training variants increased source pressure but damaged IconQA, which suggests that source gradients collide with target-critical LoRA directions.

## Mechanism Hypothesis

If the source adapter delta is decomposed into components parallel and orthogonal to the target LoRA delta, the orthogonal component should carry source/general information with less interference to the target task. This may allow a larger source injection while preserving a single static LoRA checkpoint and the same inference setting as baseline.

## Method

For each selected LoRA module:

1. Load target delta `Dt = scale_t * Bt @ At` and source delta `Ds = scale_s * Bs @ As`.
2. Remove target-aligned source update: `Ds_orth = Ds - <Ds,Dt> / ||Dt||^2 * Dt`.
3. Fuse: `Df = target_gamma * Dt + source_lambda * Ds_orth`.
4. Compress the rank-64 low-rank sum back into the target rank-32 LoRA A/B matrices using QR plus a small SVD.

The output is a normal PEFT LoRA adapter. There is no prompt gate, task label, checkpoint routing, external data, new base model, or multi-adapter inference.

## Gate

Run static checks and checkpoint generation first. Then evaluate `iconqa` before source tasks.

Promote a candidate only if:

- IconQA >= 86.20.
- Source partial scores produce required GQA <= 57.00 for Avg >= 71.05375.
- The gain is not only a post-hoc evaluation protocol change.

