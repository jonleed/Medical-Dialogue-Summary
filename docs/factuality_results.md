# Factuality results

This page reports how often the summarizers make factual errors, and how well the two automatic detectors find those errors. Every number here is rebuilt from the published benchmark in [`data/factuality_benchmark/`](../data/factuality_benchmark/) by `python -m factuality.cli rebuild`. The aggregates are in [`results/factuality/`](../results/factuality/).

Reference labels were drafted by Claude Sonnet from the blind labeling sheets and reviewed item by item by one non-clinician reviewer. The dialogues are from the MTS-Dialog validation split, which was also used to select the summarizer checkpoint.

## What is measured

Two summarizers, BioBART-large + LoRA and BART-base + LoRA, each summarize 100 validation dialogues. Two detectors look for factual errors in the summaries:

- **Rule checker.** A deterministic checker for the patient's age, sex, pronouns, gender identity and race/ethnicity. A race claim with no race statement anywhere in the dialogue is flagged as a possible insertion.
- **LLM judge.** Claude Sonnet 5.5 (`claude-sonnet-5-5`) runs in headless Claude Code with no tools beyond structured output, and no plugins, hooks or memory; an isolation probe checks this before every run. The judge sees only the dialogue, the text the summarizer could see, the requested section and the summary. It never sees rule output, the expected-fact plans, the model's identity or the construction labels.

Both detectors are scored against the reference labels, built in three steps:

1. **Plans before summaries.** For each of 50 dialogues, the drafter lists the facts the requested section should contain, from the dialogue alone, and the list is frozen before any summary is shown: 243 facts in all, 189 expected and 54 excluded ([annotation guide](factuality_annotation_guide.md)).
2. **Claim labels.** The drafter then labels every claim and every planned fact in the 100 summaries of those dialogues. The summaries appear in shuffled order with the model hidden.
3. **Split.** 15 dialogues are used to calibrate the judge rubric, and 35 are held out. Dialogues the reviewer had seen before labelling started are kept out of the held-out set.

**Headline figures use the held-out dialogues only.** The judge rubric is calibrated on the other 15 and frozen before any held-out call ([rubric calibration](../data/factuality_benchmark/rubric_revisions.md)). Intervals are 95% intervals from 2,000 bootstrap resamples of whole dialogues, with both summarizers' outputs pooled per dialogue. When a detector abstains (`unclear`) or doesn't produce a matching claim, that counts as a miss.

## Detector results on held-out dialogues

The status column follows reporting rules fixed before the held-out run. "Detects" needs at least 10 labelled positives, with precision and recall of at least 0.80. Error types with fewer than 5 positives are reported as counts only.

| Error type | Labelled positives | Judge precision (95% CI) | Judge recall (95% CI) | Rule checker recall | Status |
|---|---:|---|---|---|---|
| Expected fact omitted | 46 | 0.93 (0.81–1.00) | 0.87 (0.72–1.00) | not applicable | **judge detects omissions** |
| Unsupported claim | 16 | 0.82 (0.50–1.00) | 0.56 (0.32–0.91) | 0.00 | **judge screens for unsupported claims** |
| Age contradiction | 3 | 3 of 3 found, 0 false | — | 0 of 3 | too few positives |
| Sex-reference inconsistency | 4 | 3 of 4 found, 1 false | — | 0 of 4 | too few positives |
| Sex contradiction | 1 | 0 of 1 | — | 0 of 1 | too few positives |
| Pronoun contradiction | 0 | 1 false flag | — | — | no positives |
| Race contradiction or inserted race | 0 | none flagged | — | none flagged | no positives |

- **The 3 held-out age contradictions are all relatives' ages**, such as a parent's age at death. The rule checker looks only at the patient's own age, so it can't catch them.
  - Matching pairs claims on the same summary span even when the labels name a different subject or fact type, and counts those differences separately. 36 matched held-out claims differ only in subject.
- **On the calibration dialogues, the rule checker finds all 6 patient-age contradictions** with no false flags. These are the errors it was designed around, so this confirms behavior rather than estimating it independently.
- **Unsupported claims are the judge's weakest category.** It abstains or produces no matching claim for 31% of them. 1 unsupported flag falls on a claim the reference labels don't split out separately, and isn't counted as a false positive.
- **Omission scores cover the facts in the plans.** The judge builds its own list of expected facts, and facts it lists that aren't in a plan aren't scored either way.
- **The rule checker doesn't look for unsupported claims or omissions.** Its 0.00 recall there reflects its scope.

**Trial stability.** Claude Code has no temperature setting, so each held-out summary is judged twice, in separate calls. A claim keeps its label only when both trials agree; otherwise it becomes `unclear`. 29 of the 70 summaries differ between trials in how claims are split or worded. On every error decision that lines up with a labelled claim, the two trials agree.

## Summarizer error rates

These rates use the 46 randomly sampled dialogues. The 4 dialogues added because they had automatic flags are left out.

| Summarizer | Patient demographic error | Contradicted claims | Unsupported claims | Expected facts omitted |
|---|---|---|---|---|
| BioBART-large + LoRA | 5 / 46 summaries = 11% (2–20%) | 18 / 226 = 8% (4–13%) | 8 / 226 = 4% (1–6%) | 40 / 139 = 29% (18–39%) |
| BART-base + LoRA | 13 / 46 summaries = 28% (15–41%) | 30 / 232 = 13% (8–19%) | 19 / 232 = 8% (4–12%) | 46 / 148 = 31% (23–40%) |

A patient demographic error is a contradicted or unsupported claim about the patient's age, sex, gender identity, pronouns or race/ethnicity. Claims about relatives aren't counted.

**Output validity.** All 200 fine-tuned outputs are real summaries. Without an adapter, BART-base echoes its prompt and the dialogue in all 100 outputs, so it's reported as invalid output rather than scored for factuality.

## Race and ethnicity

**No inserted race appears anywhere:**

- **Real summaries:** across all 200 fine-tuned summaries, race appears only where the dialogue states it. 3 dialogues state the patient's race, and both models repeat it correctly.
- **Held-out labels:** none of the held-out summaries has a race error.
- **Race removed:** when the race statement is removed from those 3 dialogues, neither model adds a race (6 of 6).

**Race added to 97 dialogues** (rotated across 8 groups):

- **Summarizers:** BioBART-large carries the added race into the summary 31 times and leaves it out 66 times, never giving a wrong race. BART-base + LoRA carries it 44 times, leaves it out 52 times, and gives the wrong race once. Per-group counts are in [`counterfactual.json`](../results/factuality/counterfactual.json). With about 12 dialogues per group, they're too small to support a fairness claim.
- **Judge:** checked against the added race, it labels the summary's race claim correctly 75 of 76 times.

## Edited dialogues

Rule-checker outcomes. The expected answer covers only the edited attribute.

| Edit | BioBART-large + LoRA | BART-base + LoRA | Judge on these cases |
|---|---|---|---|
| Patient age +7 years (10) | 9 follow the new age, 1 wrong | **3 follow, 7 wrong** (1 reverts to the original age) | 20 / 20 correct |
| Patient pronouns → feminine (4) | 3 follow, 1 leaves them out | 2 follow, 2 leave them out | 5 / 5 correct |

Few dialogues meet the eligibility rules for the age and pronoun edits, so these sets are small. Pronoun swaps that would leave a masculine reference to the patient in the dialogue are rejected.

## Scope

- **Labels.** Plans and labels are drafted by Claude Sonnet, in a separate isolated session that sees only the labeling sheets, and reviewed item by item by one non-clinician reviewer. There is no independent second label, and the drafter and the LLM judge are both Claude models. Relevance and claim-splitting decisions follow the [annotation guide](factuality_annotation_guide.md).
- **Calibration.** The judge rubric is revised once on the 30 calibration summaries, so calibration figures aren't independent estimates.
- **Matching.** Two matching rules were settled while the held-out set was first scored: claims on the same span match across subject labels, and edited-dialogue summaries without the edited attribute aren't scored as judge misses. Without the first rule, age contradictions would read 0 of 3 and unsupported-claim recall 0.50.
- **One judge family.** No second judge model family is run.
- **Data.** These validation dialogues were used for model selection. The MTS-Dialog test splits aren't used.
- **Not for patient care.** Nothing here is a clinical recommendation.

## Reproducing and extending

```bash
# Rebuild every number on this page from the published benchmark (offline, a few seconds).
.venv/bin/python -m factuality.cli rebuild --check

# Summarize the benchmark dialogues with a model declared in factuality.toml.
.venv/bin/python -m factuality.cli generate --model biobart-large-lora
.venv/bin/python -m factuality.cli generate --model biobart-large-lora --counterfactual

# Score any model's summaries: validity, rule checker and edited dialogues, plus the frozen judge.
.venv/bin/python -m factuality.cli evaluate --model my-model --summaries summaries.jsonl \
  --counterfactual-summaries summaries-counterfactual.jsonl --no-judge
```

**Deterministic:** generation (fixed decoding settings, pinned model revisions), the rule checker, sampling and scoring (fixed seeds). Regenerating benchmark summaries at the pinned revisions reproduces the published summaries exactly.

**Not deterministic:** the judge. A new judge run won't reproduce the published judge outputs word for word, so those outputs are published as the record. Judge runs (dropping `--no-judge`) need a signed-in Claude Code CLI and `python -m factuality.cli probe --confirm-terms`. A new model's judge-based rates are reported next to the judge's held-out precision and recall above, but only when its judge setup (model, Claude Code version, flags, rubric and schema) matches the published freeze. With `--accept-judge-drift`, the judge runs but those reliability figures are left out.
