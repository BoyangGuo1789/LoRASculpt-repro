# Source Adapter Delta Fusion

Goal: exceed the reproduced LoRASculpt baseline by at least one average point while preserving the IconQA target adapter.

This branch is a pivot after TP-SA-MIX failed its IconQA gate. Instead of jointly retraining the target adapter, it trains a source-only OKVQA LoRA on the prepared OKVQA train split, then injects a very small fraction of that adapter into the current target-safe `gamma090` baseline with post-hoc dense LoRA fusion.

## Constraints

- No new datasets, pretrained models, paid APIs, or private data.
- OKVQA train is already present under the project data path and is separate from the OKVQA val evaluation split.
- Baseline and existing eval defaults are unchanged.
- Every completed version is recorded in `results.csv`, committed, and pushed to GitHub.

## Gate

Run smoke first:

- OKVQA train JSON builds successfully.
- 20-step source adapter train has finite loss and writes an adapter.
- Fusion produces finite tensors and high retained energy.

Then run partial eval on `iconqa,okvqa,ocrvqa,textvqa`.

Promote to GQA/full only if:

- IconQA >= 86.20.
- Required GQA for target Avg >= 71.05375 is <= 57.00.
- OKVQA is meaningfully above the gamma090 baseline, not just noise.

## Initial Candidates

- `okvqa_v_mid_l002`
- `okvqa_qv_mid_l002`
- `okvqa_v_mid_l005`
- `okvqa_qv_mid_l005`
- `okvqa_qkvo_mid_l002`
