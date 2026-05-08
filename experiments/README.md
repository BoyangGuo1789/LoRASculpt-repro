# Experiment Ledger Index

This directory records every completed experiment version that affected the LoRASculpt reproduction and plus-one search. Each completed version should be committed and pushed so it can be traced or rolled back.

## Success Criteria

- Exact reproduced baseline average: `70.05375`.
- Requested plus-one target: `71.05375`.
- Current best recorded policy: `task_aware_lora_gate`, `Avg=73.6825`.

## Ledgers

| Directory | Purpose | Current verdict |
|---|---|---|
| `task_aware_gamma/` | Dynamic task-aware routing between target LoRA and base model. | Passes plus-one target with `Avg=73.6825`; not a static checkpoint. |
| `adadare_gamma/` | Paper-default AdaDARE-gamma post-hoc fusion on target LoRA. | Fails target gate; IconQA drops to `38.47`. |
| `spider/` | Audited SPIDER training integration. | Blocked by DeepSpeed gradient access timing. |
| `source_adapter_fusion/` | Train OKVQA source adapter and inject small source deltas into target LoRA. | Target-safe but source lift is too small. |
| `tp_samix/` | Target-preserved source-anchor mixed training with PCGrad variants. | Smoke can pass; longer checkpoints collapse IconQA. |
| `samix_delta_fusion/` | Post-hoc SA-MIX source-delta fusion into target LoRA. | Target-safe but source lift is too small. |
| `samix_plus1/` | Direct mixed IconQA/COCO training. | Source signal hurts target too much. |
| `mbldf_plus1/` | Multi-branch LoRA delta fusion and mask-inspired post-hoc candidates. | Target-safe variants do not recover enough source performance. |

## Best Result Snapshot

| Metric | Score |
|---|---:|
| IconQA | 86.29 |
| OKVQA | 57.99 |
| OCRVQA | 66.15 |
| GQA | 61.93 |
| TextVQA | 58.23 |
| SourceAvg | 61.0750 |
| Avg | 73.6825 |

## Interpretation

The static checkpoint search consistently showed one failure pattern: target-preserving edits only move source scores by a small amount, while stronger source-preserving edits damage IconQA. The successful task-aware gate avoids this tradeoff by activating the target LoRA only for the target task and falling back to the base model for source/general tasks.

## Reproducibility Rule

For every completed experiment version:

1. Record the result in the relevant `results.csv`.
2. Copy compact JSON metrics or manifests into the corresponding experiment folder, using `git add -f` when needed because JSON files are ignored globally.
3. Commit the version with an `exp(...)` or `fix(...)` prefix.
4. Push the branch to GitHub before starting the next experiment version.
