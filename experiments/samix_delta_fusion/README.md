# SA-MIX Delta Fusion

Goal: reach `Avg >= 71.05375`, one point above the reproduced LoRASculpt baseline average.

## Rationale

The direct SA-MIX full trainings improved parts of the source side but damaged the IconQA target and OKVQA:

- `samix_coco1500`: OCRVQA `57.10`, TextVQA `51.41`, IconQA `84.96`, OKVQA `50.17`
- `samix_coco3000`: OCRVQA `55.20`, TextVQA `52.16`, IconQA `84.33`, OKVQA `51.49`

This suggests the source-anchor signal is real but too destructive when used as the entire adapter. This experiment keeps the current target-safe post-hoc base (`gamma_base=0.90`) and injects only small SA-MIX deltas into selected early/mid modules.

## Gate

Run partial eval first on `iconqa,okvqa,ocrvqa,textvqa`.

Promote to GQA only if both are true:

- IconQA is not materially below the current best target-safe range (`>= 86.20`).
- The required GQA for `Avg >= 71.05375` is plausible (`<= 57.0`).

## Candidate Batch 1

- `samix1500_v_mid_l005`: inject the stronger OCR candidate into early/mid `v_proj` with lambda `0.05`.
- `samix1500_vo_mid_l005`: inject the stronger OCR candidate into early/mid `v_proj/o_proj` with lambda `0.05`.
- `samix3000_v_mid_l005`: inject the stronger TextVQA candidate into early/mid `v_proj` with lambda `0.05`.
- `samix1500_mlp_mid_l005`: inject the stronger OCR candidate into early/mid MLP with lambda `0.05`.

## Paths

Exact baseline base:

```text
/data/guoboyang/LoRa-Projects/LoRASculpt-repro/checkpoints/llava-v1.5-7b-iconqa_txt_official_issue2-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1
```

SA-MIX bases:

```text
/data/guoboyang/LoRa-Projects/LoRASculpt-repro/checkpoints/llava-v1.5-7b-lorasculpt-samix-coco1500-iconqa-r32
/data/guoboyang/LoRa-Projects/LoRASculpt-repro/checkpoints/llava-v1.5-7b-lorasculpt-samix-coco3000-iconqa-r32
```

Output root:

```text
/data/guoboyang/LoRa-Projects/LoRASculpt-repro/checkpoints
```

Use `scripts/v1_5/posthoc/run_samix_delta_fusion_batch.sh` so generated checkpoint directory names contain both `llava` and `lora`; the LLaVA eval loader uses the directory basename to select the LoRA loading path.
