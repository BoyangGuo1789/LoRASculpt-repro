# SPIDER Experiments

SPIDER is tracked here as a high-risk target-preserving alternative after OKVQA source-adapter delta fusion proved source-safe but too weak.

The first action is only a 1-step smoke test before any full training. The smoke uses `tune_decoder_layer=1` instead of the audited `last2` default to reduce memory and checkpoint size. A full run must not start until a smoke run completes with finite loss and a saved checkpoint.

## Stop Criteria

- Any Deepspeed gradient access/runtime error in the 1-step smoke stops SPIDER until the trainer is patched.
- Any smoke OOM stops the current microbatch/tuned-layer setting and requires a smaller config.
- Full training is allowed only after smoke passes and the failure mode is not a trainer integration issue.
