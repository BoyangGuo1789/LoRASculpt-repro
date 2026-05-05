# SPIDER Source Audit

- Audit time: 2026-05-05T12:03:03+08:00
- Method: SPIDER, "Learn from Downstream and Be Yourself in Multimodal Large Language Models Fine-Tuning"
- Priority used: official GitHub code/README/issues, then paper.

## Official Sources Checked

- GitHub repository: https://github.com/WenkeHuang/SPIDER
- Local audited clone: /data/guoboyang/LoRa-Projects/LoRASculpt-repro/external_sources/SPIDER
- Audited commit: 197c93454560095ed357e1ebb318c7b2022921fc 2025-09-24T09:11:47+08:00 Update README.md
- OpenReview paper: https://openreview.net/forum?id=FKqmIAnkrb
- Issue #1: https://github.com/WenkeHuang/SPIDER/issues/1
- Issue #2: https://github.com/WenkeHuang/SPIDER/issues/2

## Implementation Evidence

- Main method code exists in official repository at Code/train/SPIDER.py.
- Official train.py adds tune_decoder_layer to tune the last N decoder layers together with mm_projector.
- Issue #1 confirms last-layer tuning is easy to miss in the original command surface.
- Issue #2 reports OOM even under an 8xA100 80GB ZeRO3-offload setting, so this reproduction must run max_steps=1 smoke tests before long training.

## Adopted Recommended Setting

- Model: LLaVA-1.5-7B, using our local base at /data/guoboyang/LoRa-Projects/LoRASculpt-repro/models/llava-v1.5-7b-ft.
- Target datasets: iconqa_txt and coco from this project.
- Source eval datasets: OKVQA, OCRVQA, GQA, TextVQA from this project.
- Training method: official SPIDER trainer logic from Code/train/SPIDER.py.
- Tuned modules: mm_projector plus last 2 LLM decoder layers via tune_mm_mlp_adapter=True and tune_decoder_layer=2.
- Learning rate: 2e-4 for LLM trainable layers, 2e-5 for mm_projector.
- Epochs: 5.
- Batch: global 16 by default, implemented as per-device 2 on 8 GPUs; micro-batch can be adjusted for OOM while preserving intended global batch via gradient accumulation.

## Deviations / Risks

- The full SPIDER experiment may be memory-heavy; the script is implemented, but complete runs should start only after smoke success.
- If local GPU availability differs from the paper/author environment, only micro-batch, gradient accumulation, or ZeRO/offload settings should be changed.
