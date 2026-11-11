# Clinical summary factuality benchmark

Data for rebuilding this repository's factuality results and for scoring other summarizers on the same task.

## Contents

| File | What it holds |
|---|---|
| `cases.jsonl` | 100 MTS-Dialog validation dialogues with the requested note section and the reference note |
| `summaries.jsonl` | Summaries from biobart-large-lora, bart-base-lora, bart-base, with the text each model could see and model provenance |
| `counterfactual_cases.jsonl` | Edited dialogues (age, pronoun, race), each with its exact edits and the expected answer |
| `counterfactual_summaries.jsonl` | Summaries of the edited dialogues |
| `plans.jsonl` | Expected facts for each sampled dialogue's section, written from the dialogue alone before any summary was read |
| `labels.jsonl` | 100 human claim-level labels (supported, contradicted, unsupported, unclear) and present/omitted judgments for every planned fact |
| `splits.json` | Calibration / held-out assignment and how each dialogue was sampled |
| `judge_outputs.jsonl` | The LLM judge's validated output for every summary it reviewed, by split and trial |
| `judge_config.json` | Judge model, Claude Code version, flags, and rubric/schema hashes |

Spans are 0-based, end-exclusive character offsets into the text they name. The judge rubric is `factuality/prompts/judge.txt`, and the annotation rules are in `docs/factuality_annotation_guide.md`.

## Labels

All plans and labels were written by one non-clinician reviewer (named in each record). They aren't clinical gold standard.

## License and attribution

Dialogues, reference notes and quoted spans come from MTS-Dialog (Ben Abacha et al., *An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters*, EACL 2023; https://github.com/abachaa/MTS-Dialog), licensed under CC BY 4.0. Summaries, plans, labels and judge outputs are derived additions, released under the same license. The dialogues are a research dataset, not real patient records. Nothing here is medical advice.
