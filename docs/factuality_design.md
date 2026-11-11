# Factuality evaluation: how it works

The framework measures factual errors in clinical note summaries:
- contradicted or unsupported claims about the patient's age, sex, gender identity, pronouns and race/ethnicity;
- unsupported claims in general;
- omissions of facts the requested note section should contain.

It's built to answer two questions: how often a summarizer makes these errors, and how far an automatic detector can be trusted to find them. Results are in [factuality_results.md](factuality_results.md).

## Pipeline

```
benchmark dialogues ─► summaries ─► validity gate ─┬─► rule checker ─────────┐
                                                   ├─► LLM judge ────────────┼─► scoring against reference labels
edited dialogues ───► summaries ─► validity gate ──┘   (isolated, frozen)    │     and construction gold
expected-fact plans ─► claim labels          ─────────────────────────────┘
```

Each stage writes hash-bound files, and later stages never modify earlier ones.

## Components

### Validity gate (`factuality/validity.py`)

Each output is classed as one of:
- `prompt_echo`: it contains the prompt prefix, or for outputs of 20 words or more, at least 60% of its 8-grams appear verbatim in the dialogue;
- `empty`;
- `degenerate`: for outputs of 20 words or more, one 4-gram makes up at least 30% of all 4-grams;
- `summary`.

Only `summary` outputs are scored for factuality, and the validity counts are reported per model.

### Rule checker (`factuality/rules.py`, `demographics.py`, `race.py`)

The checker extracts patient-attributed age, sex, pronoun, gender-identity and race/ethnicity claims from a summary and compares them with explicit evidence in the dialogue. Each claim gets a proposed label: supported, contradicted, unsupported or unclear.

- **Race evidence.** Race comes only from explicit statements about the patient, from the patient or the clinician, or from a confirmed question. Clinical terms such as "white blood cell" or "black stools" are excluded. Names, language, nationality and location are never treated as race evidence.
- **Inserted race.** A race claim is `unsupported` (`race_inserted`) when no race term or proxy cue appears anywhere in the dialogue.
- **Sex-reference conflicts.** A sex claim that conflicts with the patient's pronouns in the dialogue is flagged as a sex-reference inconsistency, separate from sex contradictions. Honorifics alone stay `unclear`.
- **Visibility.** Every label has a full-source view and a model-visible view, so errors whose evidence was cut off by the summarizer's 512-token input window are reported as `truncated`.

### Edited dialogues (`factuality/counterfactual.py`)

Real demographic errors are rare, so the framework makes them on purpose. Each edit is recorded as exact offsets into the original dialogue, can be replayed, and is placed inside the 512-token window. Edits that would leave the dialogue inconsistent are rejected.

| Family | Edit | Expected summary behavior |
|---|---|---|
| Race added | A race question and answer at the start of the dialogue | Copy the race, or leave it out |
| Race removed | The race statement deleted | State no race |
| Age shift | Patient age + 7 years | Follow the new age |
| Pronoun swap | Patient pronouns and honorifics masculine → feminine | Follow the swap (rejected if any masculine reference survives) |

### Reference labels (`factuality/sampling.py`, `annotate.py`)

- **Sample.** Seeded selection (seed `20261109`): 40 random dialogues, plus up to 10 flagged by the rule checker, plus random top-ups so that 35 dialogues are held out. Both models' summaries of a dialogue stay in the same split, and dialogues the reviewer saw before labelling are kept in calibration.
- **Plans.** A source-only page lists each dialogue and its requested section. Claude Sonnet drafts expected facts with exact source spans in an isolated session that sees only this page, and the plans are frozen before any summary is shown. The reviewer checks every plan.
- **Labels.** A second page shows the summaries in shuffled order with no model identity, rule output or judge output. The same drafter labels every atomic claim and marks each planned fact present, omitted or unclear, and the reviewer checks every item. The import step validates every span against the text it quotes.

### LLM judge (`factuality/judge.py`, `judge_run.py`, `claude_code_backend.py`)

- **Payload.** An explicit allowlist: opaque ID, dialogue, model-visible input, truncation flag, requested section and summary.
- **Output.** A claim inventory and a section-fact inventory, with exact quotes. Quotes that don't occur verbatim fail validation and are recorded as failed assessments.
- **Isolation.** Each call is one headless Claude Code process in an empty directory: no tools beyond structured output, no MCP servers, no slash commands, and only project settings. An isolation probe runs a canary call and fails on any tool, plugin, skill, hook event or extra context.
- **Trials and freezing.** The rubric is calibrated on the calibration split and frozen (rubric hash, schema hash, model, CLI version, flags). Held-out summaries are judged twice, and disagreement between trials makes a claim `unclear`.
- **Budget.** A file-locked ledger caps the calls at 1,000 and never re-sends a completed job. Calls stopped by a usage limit or sign-in failure can be re-sent after `judge-unblock`.

### Scoring (`factuality/metrics.py`)

- **Matching.** Detector claims are matched to reference claims on overlapping summary spans. Differences in subject or fact-type labels on the same span count as convention differences, not errors.
- **Denominators.** Abstentions and unmatched reference claims count as misses. Detector claims the reference labels don't split out are reported separately and kept out of precision.
- **Intervals.** 95% intervals come from 2,000 bootstrap resamples of whole dialogues.
- **Reporting rules,** fixed before held-out scoring:
  - "detects" needs at least 10 positives with precision and recall of at least 0.80;
  - fewer than 5 positives are reported as counts.

## Benchmark and reuse

- **Benchmark data.** [`data/factuality_benchmark/`](../data/factuality_benchmark/) holds the cases, summaries, edited dialogues, plans, labels, splits and judge outputs, as described in its datasheet. `python -m factuality.cli rebuild --check` recomputes every published aggregate from it, and CI runs that check.
- **Models.** [`factuality.toml`](../factuality.toml) declares each summarizer with a pinned base-model revision (and adapter revision) and one of three generators:
  - `flask`: this repository's serving route;
  - `hf`: any Hugging Face seq2seq model;
  - `command`: an external program that reads cases as JSONL and prints predictions.
- **New models.** `generate` writes a model's summaries, and `evaluate` scores them. A new model gets the deterministic checks, edited-dialogue outcomes and, optionally, judge-based rates reported next to the judge's measured held-out precision and recall.
- **Environment.** [`requirements-lock.txt`](../requirements-lock.txt) records the full environment behind the published results.
