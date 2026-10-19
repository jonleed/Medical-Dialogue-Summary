# Dataset choice for this project — October 18, 2026

MTS-Dialog's official evaluation data **can be used**. A separate dataset is useful for testing transfer; it is not a requirement for a valid in-domain test.

The checked-in [training recipe](../train_lora.py) loads the 1,201-row training file for gradient updates and the 100-row validation file for per-epoch evaluation and best-checkpoint selection. Validation is development data. The current 300 saved summaries come from those same 100 validation dialogues. The notebook and app also load Test Set 1, so a file being named “test” does not establish that it has never influenced this project's decisions. No complete historical checkpoint-training manifest is available to prove every past use.

| Resource | What it provides | Best role here | Limits |
|---|---|---|---|
| MTS-Dialog Test Set 2 | 200 short dialogue/section-note pairs, matching our task | Primary final in-domain summarizer evaluation after freezing models, prompts and evaluator | Audit historical use and remove overlaps with training/development first |
| MTS-Dialog correlation study | 400 generated summaries with manual factual precision, recall, hallucination and omission scores | Test agreement of evaluator scores with existing expert judgments | The underlying 100 validation dialogues are already familiar; these are other systems' outputs, not labels for our 300 outputs; released labels are summary scores, not complete atomic spans |
| ACI-Bench | 207 full encounter/note pairs: 67 train, 20 validation, three 40-case test sets | External transfer evaluation on longer encounters | Different task length; current 512-token input limit would truncate many encounters, requiring separate truncation-aware reporting; some speaker tags are known to be swapped |
| MedVAL-Bench public dialogue2note subset | 85 ACI-Bench-derived input/output pairs with physician error assessments and risk grades | Calibrate and test the evaluator against actual physician assessments | ACI-Bench and MedVAL dialogue2note are related data, not independent corpora; risk grades are not exhaustive claim/omission gold |

Official sources: [MTS-Dialog release](https://github.com/abachaa/MTS-Dialog), [MTS-Dialog paper](https://aclanthology.org/2023.eacl-main.168/), [ACI-Bench release](https://github.com/wyim/aci-bench), [MedVAL-Bench card](https://huggingface.co/datasets/stanfordmimi/MedVAL-Bench). MTS and ACI are publicly released under CC BY 4.0. The public MedVAL release has an MIT card license; respect any underlying source terms. No restricted MIMIC/PhysioNet records were accessed.

## Local overlap audit

The [split audit](../results/factuality/semantic-study-2026-10-18/mts_split_audit.json) pins the local dataset revision and all CSV hashes. Comparing lowercased input text with punctuation/whitespace removed found two unique normalized dialogue strings shared between training and validation, two between training and Test Set 1, and one between training and Test Set 2. These are overlap signals, not proof that entire clinical cases are duplicates; generic snippets can recur. A conservative final cohort should exclude matching input strings and check near-duplicates/shared source notes. Repeated short reference phrases alone do not establish leakage.

The public MedVAL dialogue inputs have zero exact normalized matches against either the 1,201 training rows or the 100 project validation sources. That does not rule out paraphrases or exposure during general model pretraining.

## How to use clinician-written references

References identify expected clinical content and support omission analysis. Match meanings, not exact wording; the dialogue remains the evidence for patient-specific claims. Reference notes are not automatically exhaustive, and their contents can be more specific than the transcript. The MTS paper describes simulated conversations written from existing clinical notes by medically trained annotators, including former medical scribes; it does not establish that every released item was independently written and verified by a physician.

To measure the evaluator itself, use generated notes with known expert error judgments or controlled edits with known targets. A reference note alone cannot certify the evaluator's sensitivity, false-positive rate, or extraction completeness. Existing expert annotations reduce the need to recruit a new annotation team, but do not constitute new expert review of our model's outputs.

Recommendation: keep MTS validation for development, freeze a decontaminated MTS Test Set 2 cohort for final in-domain evaluation, use the published MTS manual scores to check evaluator agreement, and use ACI/MedVAL as a complementary external check with their overlap disclosed. Public data can be excluded from **our fine-tuning** using training provenance and overlap audits; guaranteed absence from a frontier judge's undisclosed pretraining cannot be established.
