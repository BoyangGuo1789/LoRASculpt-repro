# AdaDARE-gamma Experiments

This ledger tracks post-hoc AdaDARE-gamma fusion on the current target-safe IconQA LoRA baseline. The target is an average score at least one point above the reproduced baseline, so every candidate must first preserve IconQA before any source/GQA compute is spent.

## Gate Policy

- Run dry-run first and record changed tensors plus observed retention.
- Run full fusion only if dry-run confirms the expected retention ratio and no load errors.
- Evaluate IconQA first. Stop immediately if IconQA is below 86.20.
- Only run OKVQA/OCRVQA/TextVQA/GQA when IconQA passes the gate.

## 2026-05-08 Result

The paper-default LoRA setting (`gamma=0.7`, `sparsity=0.9`, deterministic top-k, unit-Hessian proxy) is not target-safe on this already task-tuned IconQA adapter. It retained about 10.31% of changed delta elements but amplified selected weights enough to collapse IconQA to 38.47%, so the direction is stopped unless a much smaller effective gamma or a LoRA-space variant is explicitly designed.
