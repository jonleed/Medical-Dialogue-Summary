# Factuality evaluation v3: results

Run 2026-11-09. Code is on branch `codex/factuality-v3`. Public aggregates (counts and rates only, no clinical text) are in [`results/factuality/v3-2026-11-09/`](../results/factuality/v3-2026-11-09/). The raw dialogues, summaries, labels and judge responses stay local.

**Caveat on every number below:** single non-clinician annotator, 35 held-out dialogues. The dialogues come from the MTS-Dialog validation split, which was already used to select the summarizer checkpoint.

## What was measured

Two summarizers were evaluated, BioBART-large + LoRA and BART-base + LoRA, on 100 validation dialogues each. Two detectors look for factual errors in their summaries:

- **Rules:** `patient_demographics_v3`, a deterministic checker. It covers patient age, sex, pronouns, gender identity and race/ethnicity. Race claims that appear nowhere in the source are flagged as possible insertions.
- **Judge:** Claude Sonnet 5.5 (`claude-sonnet-5-5`), run in headless Claude Code 2.1.311 with no tools beyond schema output, and no plugins, hooks or memory. An isolation probe verified this before each run. The judge received only the dialogue, the model-visible input, the requested section and the summary. It never saw rule output, plans, model identity or construction labels.

Both detectors are scored against the owner's labels:

1. **Blind plans.** Before seeing any summary, the owner wrote and froze a plan for each of 50 dialogues: the facts the requested section should contain (243 facts, 189 expected, 54 excluded).
2. **Labels.** The owner then labelled every claim and every planned fact in the 100 summaries of those dialogues. The summaries were shown in shuffled order with the model hidden.
3. **Split.** 15 dialogues were used for calibration and 35 held out. The 8 dialogues the owner had already seen during the audit were barred from held-out.

**Headline numbers use held-out only.** The judge rubric was revised once on calibration (two rounds; see the rubric log) and frozen before any held-out call.

Uncertainty intervals are 95% intervals from 2,000 bootstrap resamples of whole dialogues, with both summarizers' outputs pooled per dialogue. Where a detector abstained (`unclear`) or didn't match a labelled claim, that counts as a miss.

## Held-out detector results

Each status is fixed by a reporting rule written before the held-out run. "Detects" requires at least 10 labelled positives with precision and recall of at least 0.80. Error types with fewer than 5 positives are reported only as counts.

| Error type | Labelled positives | Judge precision (95% CI) | Judge recall (95% CI) | Rules recall | Status |
|---|---:|---|---|---|---|
| Expected fact omitted | 46 | 0.93 (0.81–1.00) | 0.87 (0.72–1.00) | not applicable | **judge detects omissions** |
| Unsupported claim | 16 | 0.82 (0.50–1.00) | 0.56 (0.32–0.91) | 0.00 | **judge screens for unsupported claims** |
| Age contradiction | 3 | 3 of 3 found, 0 false | — | 0 of 3 | insufficient positives |
| Sex-reference inconsistency | 4 | 3 of 4 found, 1 false | — | 0 of 4 | insufficient positives |
| Sex contradiction | 1 | 0 of 1 | — | 0 of 1 | insufficient positives |
| Pronoun contradiction | 0 | 1 false flag | — | — | no positives |
| Race contradiction / inserted race | 0 | none flagged | — | none flagged | no positives |

**Notes:**

- **The 3 held-out age contradictions are all relatives' ages** (for example, a parent's age at death). The rules check only the patient's own age, so they can't catch these.
  - The labels record these claims with the page's default subject, `patient`, while the judge labelled them `relative`. Scoring matches claims on the same summary span even when the subject or fact type differs, and counts those differences separately.
  - Across all matched held-out claims, 36 differ only in subject.
  - On the **calibration** split, the rules found all 6 patient-age contradictions with no false flags. This is a fair test of the rules: they were frozen before any labels existed and were never adjusted.
- **Unsupported claims** are the judge's weakest category. In 31% of these cases it abstained or didn't produce a matching claim. 1 unsupported flag fell on a claim the owner hadn't split out separately; it isn't counted as a false positive.
- **The rules flag no unsupported or omission errors by design.** Their 0.00 recall on unsupported claims reflects scope, not failure.

**Trial stability.** Claude Code exposes no temperature setting, so each held-out summary was judged twice. Claims are kept only when both trials agree; otherwise they become `unclear`. The two trials were separate calls. 29 of 70 summaries differed between trials in how claims were split or worded. On every error decision that lines up with a labelled claim, the two trials agreed.

## Summarizer error rates

These rates come from the 46 randomly sampled dialogues: 40 random, plus 6 random dialogues added to bring held-out up to 35. The 4 enrichment dialogues are excluded.

| Summarizer | Contradicted claims | Unsupported claims | Expected facts omitted |
|---|---|---|---|
| BioBART-large + LoRA | 18 / 226 = 8% (4–13%) | 8 / 226 = 4% (1–6%) | 40 / 139 = 29% (18–39%) |
| BART-base + LoRA | 30 / 232 = 13% (8–19%) | 19 / 232 = 8% (4–12%) | 46 / 148 = 31% (23–40%) |

**Output validity.** All 200 fine-tuned outputs are real summaries. All 100 BART-base outputs without an adapter echo the prompt and dialogue, so BART-base is reported only as an echo rate, not as a factuality row.

## Race/ethnicity

**No inserted race was found anywhere:**

- **Real summaries:** across all 200 fine-tuned summaries, the rules flagged no race claim without source support. 3 dialogues state the patient's race, and both models repeat it correctly in all 3.
- **Held-out labels:** the owner labelled no race error in held-out.
- **Race removed:** when the race statement was edited out of those 3 dialogues, neither model added a race (6 of 6 faithful).

**Injected race** (97 dialogues, rotated across 8 groups):

- **Summarizers:** BioBART-large carried the injected race into the summary 31 times and left it out 66 times, with no wrong race. BART-base + LoRA carried it 44 times, left it out 52 times and gave the wrong race once. Per-group counts are in the aggregate. With about 12 dialogues per group they're underpowered, and this isn't a fairness claim.
- **Judge:** checked against the injected race, it labelled the summary's race claim correctly 75 of 76 times.

## Counterfactual edits

These are rule-side outcomes. Construction gold covers only the edited attribute.

| Edit | BioBART-large + LoRA | BART-base + LoRA | Judge on these cases |
|---|---|---|---|
| Patient age +7 years (10) | 9 followed the new age, 1 wrong | **3 followed, 7 wrong** (1 reverted to the original age) | 20 / 20 correct |
| Patient pronouns → feminine (4) | 3 followed, 1 left out | 2 followed, 2 left out | 5 / 5 correct |

The age and pronoun sets are small because few dialogues met the eligibility rules. 3 pronoun swaps were discarded after review because masculine references to the patient remained.

## Limitations and provenance

- **One annotator, not a clinician.** No inter-annotator agreement is available. Relevance and atomicity decisions follow the written [plan guide](factuality_v3_plan_guide.md).
- **Label provenance.** Earlier exports of the 70 held-out items carried the reviewer string "Claude draft labels (pending review by Jonathan Leeder)", which another session had assigned. The owner attests that they authored and reviewed all 100 items. The reviewer field was corrected, and the label content is unchanged. The record is kept locally in `annotation_provenance.json`.
- **Prompt revision.** The judge rubric was revised once on 30 calibration summaries: contradicted vs. unsupported, claim granularity, and the quote rule. It was then frozen. Calibration figures are not independent estimates.
- **No second judge.** No second judge model family was run. It was planned as an optional sensitivity check and dropped after its API key was rejected.
- **Development data.** These validation dialogues were already used for model selection. The MTS-Dialog test splits were not used.
- **Not for patient care.** No patient-care recommendations follow from this study.

## Reproduction

Raw artifacts are local. With them, each stage runs through `python -m factuality.v3_cli <stage>`: validity, screen, cf-build, cf-score, sample, plans, freeze-plans, label-packet, import-labels, probe, judge, judge-freeze, score. The offline test suite needs no artifacts:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
```
