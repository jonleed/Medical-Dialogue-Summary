# Clinical summary factuality benchmark

Data for rebuilding this repository's factuality results and for scoring other summarizers on the same task.

## Contents

| File | What it holds |
|---|---|
| `cases.jsonl` | 100 MTS-Dialog validation dialogues with the requested note section and the reference note |
| `summaries.jsonl` | Summaries from biobart-large-lora, bart-base-lora, bart-base, with the text each model could see and model provenance |
| `counterfactual_cases.jsonl` | Edited dialogues (age, pronoun, race), each with its exact edits and the expected answer |
| `counterfactual_summaries.jsonl` | Summaries of the edited dialogues |
| `plans.jsonl` | Expected facts for each sampled dialogue's section, drafted from the dialogue alone before any summary was shown |
| `labels.jsonl` | 100 reviewed claim-level labels (supported, contradicted, unsupported, unclear) and present/omitted judgments for every planned fact |
| `splits.json` | Calibration / held-out assignment and how each dialogue was sampled |
| `judge_outputs.jsonl` | The LLM judge's validated output for every summary it reviewed, by split and trial |
| `judge_config.json` | Judge model, Claude Code version, flags, and rubric/schema hashes |

Spans are 0-based, end-exclusive character offsets into the text they name. The judge rubric is `factuality/prompts/judge.txt`, and the annotation rules are in `docs/factuality_annotation_guide.md`.

## Labels

Plans and claim labels were drafted by Claude Sonnet in a separate, isolated session. The drafter worked from the labeling sheets and had no other context. For plans, the sheet was source-only: each dialogue and its requested section, with plans frozen before any summary was shown. For labels, the sheet showed the summaries in shuffled order with no model identity, rule-checker output or judge output. The drafter listed the facts each section should contain, labeled every atomic claim as supported, contradicted, unsupported or unclear, and marked each planned fact as present or omitted. Jonathan Leeder, a non-clinician, reviewed every item and is the reviewer of record (named in each record). The labels are not a clinical gold standard. There is a single reviewer and no independent second label. The drafter and the LLM judge are both Claude models.

## License and attribution

Dialogues, reference notes and quoted spans come from MTS-Dialog (Ben Abacha et al., *An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters*, EACL 2023; https://github.com/abachaa/MTS-Dialog), licensed under CC BY 4.0. Summaries, plans, labels and judge outputs are derived additions, released under the same license. The dialogues are a research dataset, not real patient records. Nothing here is medical advice.
