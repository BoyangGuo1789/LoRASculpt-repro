# TP-SA-MIX

Target-preserved SA-MIX starts from the exact reproduced IconQA LoRA and trains on IconQA plus COCO-caption anchors. The first implementation keeps the baseline path untouched and adds a separate `TPSAMIX` trainer with:

- baseline LoRA initialization through `--lora_start_path`
- frozen `teacher` LoRA adapter for answer-token top-k KL
- source sample weighting from `samix_source`
- LoRA L2 distance to the baseline adapter
- no inherited LoRASculpt CMR term, because the baseline-distance anchor already constrains drift and avoids the extra full-matrix allocation during teacher-student training

PCGrad is intentionally reserved for a later version after the KD/anchor smoke passes.
