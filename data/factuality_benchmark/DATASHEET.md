# Clinical summary factuality benchmark

Data for rebuilding this repository's factuality results and for scoring other summarizers on the same task.

## Contents

| File | What it holds |
|---|---|
| `cases.jsonl` | 100 MTS-Dialog validation dialogues with the requested note section and the reference note |
| `summaries.jsonl` | Summaries from biobart-large-lora, bart-base-lora, bart-base, with the text each model could see and model provenance |
| `counterfactual_cases.jsonl` | Edited dialogues (age, pronoun, race), each with its exact edits and the expected answer |
| `counterfactual_summaries.jsonl` | Summaries of the edited dialogues |
| `plans.jsonl` | Expected facts for each sampled dialogue's section, AI-drafted from the dialogue alone before any summary was shown, then human-reviewed |
| `labels.jsonl` | 100 AI-drafted, human-reviewed claim-level labels (supported, contradicted, unsupported, unclear) and present/omitted judgments for every planned fact |
| `splits.json` | Calibration / held-out assignment and how each dialogue was sampled |
| `judge_outputs.jsonl` | The LLM judge's validated output for every summary it reviewed, by split and trial |
| `judge_config.json` | Judge model, Claude Code version, flags, and rubric/schema hashes |

Spans are 0-based, end-exclusive character offsets into the text they name. The judge rubric is `factuality/prompts/judge.txt`, and the annotation rules are in `docs/factuality_annotation_guide.md`.

## Labels

Plans and claim labels are AI-drafted and human-reviewed. Claude Sonnet drafted them in a separate, isolated session from the labeling sheets, with no other context. For plans, the sheet was source-only: each dialogue and its requested section, with plans frozen before any summary was shown. For labels, the sheet showed the summaries in shuffled order with no model identity, rule-checker output or judge output. The drafter listed the facts each section should contain, labeled every atomic claim as supported, contradicted, unsupported or unclear, and marked each planned fact as present or omitted. One non-clinician human reviewer (`reviewer-a`, the project author) then checked every item against the dialogue. The files hold the reviewed labels, not the drafter's first output, so how often the reviewer changed a draft is not recorded. The reviewer saw the draft, so the labels are not an independent second opinion, and they are not a clinical gold standard. The drafter and the LLM judge are both Claude models.

In `labels.jsonl`, `reviewer` is an opaque reviewer id and `reviewer_kind: "human"` marks human-reviewed items. `status` is `partial` in every record because the export tool never writes `complete`; it does not mean a partial review. Every record has `claims_complete` and `expected_facts_complete` set and no unresolved items.

## License and attribution

Dialogues, reference notes and quoted spans come from MTS-Dialog (Ben Abacha et al., *An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters*, EACL 2023; https://github.com/abachaa/MTS-Dialog), licensed under CC BY 4.0. Summaries, plans, labels and judge outputs are derived additions, released under the same license. The dialogues are a research dataset, not real patient records. Nothing here is medical advice.
