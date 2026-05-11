# Experiment Ledger Index

This directory records every completed experiment version that affected the LoRASculpt reproduction and plus-one search. Each completed version should be committed and pushed so it can be traced or rolled back.

## Success Criteria

- Exact reproduced baseline average: `70.05375`.
- Requested plus-one target: `71.05375`.
- Current best archived method: `lora_input_gate/LoRA-IGP-Ponly`, `Avg=73.7025`.

## Ledgers

| Directory | Purpose | Current verdict |
|---|---|---|
| `lora_input_gate/` | Input-conditioned internal adaptation gating; final best variant gates only the `mm_projector` adaptation path while keeping target LoRA active. | Archived best result: `Avg=73.7025`, +3.64875 over exact baseline. |
| `task_aware_gamma/` | Dynamic task-aware routing between target LoRA and base model. | Passes plus-one target with `Avg=73.6825`; not a static checkpoint. |
| `adadare_gamma/` | Paper-default AdaDARE-gamma post-hoc fusion on target LoRA. | Fails target gate; IconQA drops to `38.47`. |
| `spider/` | Audited SPIDER training integration. | Blocked by DeepSpeed gradient access timing. |
| `source_adapter_fusion/` | Train OKVQA source adapter and inject small source deltas into target LoRA. | Target-safe but source lift is too small. |
| `tp_samix/` | Target-preserved source-anchor mixed training with PCGrad variants. | Smoke can pass; longer checkpoints collapse IconQA. |
| `samix_delta_fusion/` | Post-hoc SA-MIX source-delta fusion into target LoRA. | Target-safe but source lift is too small. |
| `samix_plus1/` | Direct mixed IconQA/COCO training. | Source signal hurts target too much. |
| `mbldf_plus1/` | Multi-branch LoRA delta fusion and mask-inspired post-hoc candidates. | Target-safe variants do not recover enough source performance. |
| `pars_lora/` | Projector-anchored rank-split LoRA inside baseline training. | Rejected after B-balanced and C-flex underperformed exact IconQA baseline. |
| `rank_path_balanced_lora/` | Rank-path balanced LoRA mask selection that keeps every low-rank path alive before filling with global score. | Rejected: q060 improves IconQA to `86.64`, but source tasks fall and Avg is `69.44875`. |
| `source_activation_null_lora/` | Training-time source-activation nulling loss that keeps one always-on LoRA but penalizes its source-example delta activations. | Active next line after RPB source-forgetting failure. |

## Best Result Snapshot

| Metric | Score |
|---|---:|
| IconQA | 86.30 |
| OKVQA | 57.81 |
| OCRVQA | 66.30 |
| GQA | 61.96 |
| TextVQA | 58.35 |
| SourceAvg | 61.1050 |
| Avg | 73.7025 |

## Interpretation

The static checkpoint search consistently showed one failure pattern: target-preserving edits only move source scores by a small amount, while stronger source-preserving edits damage IconQA. RPB adds a sharper version of this evidence: rank-path balanced support can raise IconQA (`86.64`) but still worsens source/general average. The archived best result shows that a single checkpoint can clear the plus-one target by controlling the target-specific projector adaptation path, but this line is now sealed. The next research phase should use a substantially different baseline upgrade, preferably a training-time objective, LoRA-internal module, scoring rule, or regularizer rather than continuing prompt-form gating.

## Reproducibility Rule

For every completed experiment version:

1. Record the result in the relevant `results.csv`.
2. Copy compact JSON metrics or manifests into the corresponding experiment folder, using `git add -f` when needed because JSON files are ignored globally.
3. Commit the version with an `exp(...)` or `fix(...)` prefix.
4. Push the branch to GitHub before starting the next experiment version.
