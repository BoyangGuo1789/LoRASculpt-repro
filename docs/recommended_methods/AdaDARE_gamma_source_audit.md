# AdaDARE-gamma Source Audit

- Audit time: 2026-05-05T12:03:03+08:00
- Method: AdaDARE-gamma, "Balancing Stability and Plasticity in Multi-modal LLMs through Efficient Adaptation"
- Priority used: official code search, project/paper page, supplement, paper text.

## Official Sources Checked

- CVF paper page: https://openaccess.thecvf.com/content/CVPR2025/html/Xie_AdaDARE-gamma_Balancing_Stability_and_Plasticity_in_Multi-modal_LLMs_through_Efficient_CVPR_2025_paper.html
- CVF PDF: https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_AdaDARE-gamma_Balancing_Stability_and_Plasticity_in_Multi-modal_LLMs_through_Efficient_CVPR_2025_paper.pdf
- Supplement from the local reference folder: /Users/leo/Documents/New project 2/参考论文/AdaDARE-gamma_CVPR2025/supplement.pdf
- Web/GitHub searches for AdaDARE-gamma and AdaDARE did not identify an official author GitHub repository at implementation time.

## Adopted Recommended Setting

- Model: LLaVA-1.5-7B, using our local base at /data/guoboyang/LoRa-Projects/LoRASculpt-repro/models/llava-v1.5-7b-ft.
- Target datasets: iconqa_txt and coco from this project.
- Source eval datasets: OKVQA, OCRVQA, GQA, TextVQA from this project.
- PEFT setting: LoRA rank 128, learning rate 5e-6, 5 epochs.
- LoRA alpha: 256, following the common LLaVA LoRA rank-128 convention when the AdaDARE-gamma paper specifies rank but not alpha.
- AdaDARE-gamma fusion: gamma=0.7 and sparsity=90% for the LoRA setting, matching the paper table for LoRA combination.

## Implementation Evidence

- No official AdaDARE-gamma code was found; implementation is labeled as a paper/supplement self implementation, not an official-code reproduction.
- The postprocess script applies adaptive DARE probabilities to the merged task delta and then injects gamma-scaled selected delta into the pretrained model.
- Because no released Hessian/statistics code is available, the implemented default uses unit-Hessian scoring, equivalent to score_i=abs(delta_i). This is explicitly recorded in the fused checkpoint config.

## Deviations / Risks

- The missing official code means the exact Hessian approximation may differ from the authors internal implementation.
- Full fusion loads a merged tuned model and a base model on CPU; disk and RAM are sufficient on the current server, but runtime will be non-trivial.
