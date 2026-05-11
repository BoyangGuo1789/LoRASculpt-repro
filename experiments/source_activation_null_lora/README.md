# Source-Activation Nulling LoRA

## Baseline limitation

The LoRASculpt baseline trains a single target LoRA and applies it to every input. Recent runs make the failure clearer:

- Direct source/target mixed training and TP-SA-MIX added source answer losses, but IconQA collapsed.
- Static source-delta fusion preserved IconQA only at tiny coefficients, giving too little source lift.
- RPB improved IconQA to 86.64 but degraded all source/general tasks, so rank-path support alone is not the bottleneck.

The working hypothesis is that the always-on LoRA target update is not sufficiently selective in activation space. It helps IconQA, but also produces non-trivial deltas on source/general visual-language activations.

## Method

SAN-LoRA keeps the baseline inference setting: one model, one LoRA adapter, no task gate, no checkpoint routing, and no evaluation-time LoRA disabling.

During training it uses a mixed IconQA + COCO-caption stream:

1. IconQA examples optimize the normal target language-modeling loss plus the original LoRASculpt CMR regularizer.
2. COCO/source examples do not optimize caption CE.
3. Forward hooks on selected LoRA-B modules collect the LoRA delta activations for target and source examples.
4. A source-activation nulling loss penalizes the source/target LoRA-delta energy ratio:

```text
L = L_target_ce + L_CMR
    + lambda_san * mean_m ||Delta_m(h_source)||_2^2 / stopgrad(||Delta_m(h_target)||_2^2 + eps)
```

This makes LoRA remain always active, but trains it to be approximately silent on source-style activations.

## Decision Rule

Smoke must show finite target CE, CMR, SAN ratio loss, hook count greater than zero, one prune event at step 1, and an adapter checkpoint. Full training only promotes to source/general evaluation if IconQA is at least 86.20.

Because the SAN loss is built from forward hooks on LoRA-B outputs, the source mask is intentionally kept alive until backward finishes so gradient-checkpointed recomputation follows the same hook path.

## Smoke Result

`san_ratio_l005_qkv_smoke_20260512_020358` passed:

- checkpoint: `/data/guoboyang/LoRa-Projects/LoRASculpt-repro/checkpoints/llava-v1.5-7b-lorasculpt-san-ratio-l005-qkv-r32-20260512_020358-smoke`
- log: `/data/guoboyang/LoRa-Projects/LoRASculpt-repro/logs/lorasculpt_san_l005_ratio_qkv_smoke_20260512_020358.log`
- 20 optimizer steps completed with `train_loss=0.5935190986841917`
- q/k/v hook count reached 96 on mixed target/source batches
- observed source/target delta energy ratio was about `0.91-0.98` on active SAN batches
- `adapter_model.bin` and `non_lora_trainables.bin` were written

Two implementation-only smoke failures preceded this pass: clearing the source mask before backward broke checkpoint recomputation, and disabling checkpointing caused OOM. The accepted implementation keeps the mask alive through backward and runs with gradient checkpointing enabled.
