# PARS-LoRA: Projector-Anchored Rank-Split LoRA

PARS-LoRA is the next baseline-internal research branch after sealing the PSL diagnostic. It keeps one static checkpoint and one always-active LoRA adapter at inference.

## Baseline Limitation

LoRASculpt adapts IconQA by training LoRA and `mm_projector`, then applies the baseline absolute top-k LoRA mask. The observed failure pattern is that target gains can coexist with source/general degradation, which suggests two coupled bottlenecks:

- The multimodal projector can drift away from the source/general alignment that OKVQA, OCRVQA, GQA, and TextVQA still need.
- All LoRA ranks train at the same rate, so the adapter has no explicit stable subspace to absorb low-risk shared directions while plastic ranks adapt to IconQA.

## Mechanism Hypothesis

Constrain projector drift during training with a normalized residual soft cap, and split LoRA ranks into stable and plastic subspaces that remain jointly active. The stable ranks receive a smaller gradient scale; plastic ranks keep full learning rate. A small orthogonal decorrelation term discourages both subspaces from collapsing onto the same directions.

## Method

Training objective:

```text
L = L_ce + L_cmr + L_proj + L_orth
```

Projector soft cap:

```text
d_proj = sqrt(sum ||P_i - P0_i||^2 / (sum ||P0_i||^2 + eps))
L_proj = lambda_proj * ramp(step) * relu(d_proj - tau_proj)^2
```

Rank split:

```text
stable ranks = 0..7, plastic ranks = 8..31 for rank 32
A.grad[:8] *= 0.25
B.grad[:, :8] *= 0.25
```

Mainline keeps the LoRASculpt baseline abs top-k mask by setting `MIGDIS_ENABLE=False`; no task routing, checkpoint routing, prompt routing, or LoRA-off evaluation is used.

## Promotion Gate

