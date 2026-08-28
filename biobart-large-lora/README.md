---
base_model: GanjinZero/biobart-large
library_name: peft
tags:
- lora
- peft
- medical
- summarization
---

# Clinical Note BioBART-large LoRA

LoRA adapter fine-tuned from `GanjinZero/biobart-large` on the MTS-Dialog dataset using the original project prompt and target construction.

Configuration: LoRA rank 8, alpha 64, dropout 0.01, targeting BART query/value projections. Use with `GanjinZero/biobart-large` through PEFT.

This is an educational research artifact. Generated notes require clinician review and are not clinically validated.
