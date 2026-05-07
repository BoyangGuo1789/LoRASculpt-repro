# SPIDER Experiments

SPIDER is tracked here as a high-risk target-preserving alternative after OKVQA source-adapter delta fusion proved source-safe but too weak.

The first action is only a 1-step smoke test before any full training. The smoke uses `tune_decoder_layer=1` instead of the audited `last2` default to reduce memory and checkpoint size. A full run must not start until a smoke run completes with finite loss and a saved checkpoint.

## Stop Criteria

- Any Deepspeed gradient access/runtime error in the 1-step smoke stops SPIDER until the trainer is patched.
- Any smoke OOM stops the current microbatch/tuned-layer setting and requires a smaller config.
- Full training is allowed only after smoke passes and the failure mode is not a trainer integration issue.


## 2026-05-08 Diagnostics

- Initial 1-step Deepspeed smoke failed because `safe_get_full_grad(param)` was called after `self.optimizer.step()`.
- A local diagnostic patch moved `self.optimizer.step()` after the grad-rank block, but Deepspeed still reported gradients unavailable at the same `safe_get_full_grad(param)` call before a checkpoint or metric could be produced.
- A no-Deepspeed torchrun diagnostic failed earlier because `safe_get_full_fp32_param(param)` returned `None`, so the audited trainer is coupled to Deepspeed ZeRO helper state rather than a plain PyTorch optimizer path.
- The diagnostic patch was reverted after recording the evidence. SPIDER remains blocked until its gradient access strategy is redesigned inside the actual Deepspeed backward/step lifecycle.
- Do not launch SPIDER full training from this implementation state.
