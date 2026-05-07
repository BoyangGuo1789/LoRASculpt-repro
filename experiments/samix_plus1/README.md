# SA-MIX Plus1 Experiments

SA-MIX is a source-anchor mixed training branch for the LoRASculpt Issue2 IconQA setup.

Goal:

- Exact reproduced baseline Avg: 70.05375.
- Current best Avg: 70.09625.
- Target Avg: 71.05375.

Working hypothesis:

- Prior mask-selection and post-hoc fusion attempts preserved IconQA but did not create enough source/general lift.
- The next credible way to target a +1 Avg gain is to add a bounded source-anchor signal during training.
- The only training-like source data currently available without contaminating eval sets is COCO caption train from the official Issue2 packet.

Constraints:

- Do not train on OKVQA/OCRVQA/GQA/TextVQA eval files.
- Do not introduce external datasets, new pretrained models, paid APIs, private data, or unbounded training.
- Do not change baseline scripts or default LoRASculpt behavior.

Method:

- Build a deterministic mixed LLaVA JSON from:
  - official Issue2 IconQA train: 10,000 samples
  - official Issue2 COCO caption train: configurable sample count
- Normalize image paths so one common image root can serve both datasets:
  - `image_folder=/data/guoboyang/LoRa-Projects/LoRASculpt-repro/data`
- Train normal `LoRASculpt` rank-32 LoRA with the same core hyperparameters as the reproduced baseline.

Initial variants:

| candidate | IconQA samples | COCO samples | intent |
|---|---:|---:|---|
| samix_coco1500 | 10000 | 1500 | conservative source anchor |
| samix_coco3000 | 10000 | 3000 | stronger source anchor |

Partial gate:

Run `iconqa,okvqa,ocrvqa,textvqa` first. Promote to GQA/full table only if:

- IconQA >= 86.20
- OKVQA >= 52.50
- OCRVQA >= 55.50
- TextVQA >= 52.00
- required GQA for target <= 57.00

Full success:

- Avg >= 71.05375.

Result table:

| run_name | stage | checkpoint | iconqa_train | coco_train | OKVQA | OCRVQA | GQA | TextVQA | SourceAvg | IconQA | Avg | delta_vs_baseline | promote | stop_reason | log_file | git_commit |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
