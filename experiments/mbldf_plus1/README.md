# MB-LDF Plus1 Experiment

MB-LDF is a training-free post-hoc LoRA delta fusion experiment for the LoRASculpt IconQA reproduction.

Goal:

- Current best: baseline LoRA `gamma=0.90`, `Avg=70.09625`.
- Target: `Avg >= 71.05375`.

Phase 1 uses only existing checkpoints:

- exact Issue2 baseline,
- baseline `gamma=0.90`,
- MIG-DIS V4,
- MIG-DIS V5,
- DQSS-r025g.

Fusion happens in dense LoRA delta space and is projected back to rank 32 with compact SVD. Official evaluation data and scoring are unchanged.

Results are appended to `results.csv`. Per-candidate metrics and manifests should be copied into `metrics/` and `manifests/` before each commit.
