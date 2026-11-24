# Label revisions

The reviewer labels follow one rule for the patient's gender reference: sex terms (male, female) and pronouns (he, she) are a single dimension, and evidence for it is an explicit statement about the patient, a third-person pronoun that refers to the patient, or a gendered form of address directed at the patient (sir, ma'am, miss, mister, young man). A matching claim is supported, a conflicting claim is contradicted, and a claim with no evidence is unsupported. Cues that disagree with each other, evidence about someone else, and inferences such as a pregnancy test leave the claim unclear. Names and casual address ("bud") are not evidence. Gender identity is a separate dimension.

The original labels already followed that rule almost everywhere. After the rule was fixed, every gender-reference label was compared with the rule checker, and the labels below were the exceptions. The same pass corrected two weak plan spans and two borderline fact labels. The reviewer of record checked each edit. `scripts/relabel_gender_reference.py` applies exactly these edits to the original bundle (commit `a078f7c`). Re-exporting the bundle from the local study files would restore the original labels, so apply the script to a fresh export.

The judge outputs in the benchmark came from the original judge rubric (`factuality/prompts/archive/judge_original.txt`), which said honorifics do not establish sex, gender identity or pronouns. The current rubric (`factuality/prompts/judge.txt`) carries the merged rule. The outputs are scored against the revised labels as they are, so judge-versus-human differences on honorific claims reflect the rubric change, not judge error. A judge rerun under the current rubric needs a new freeze.

| Split | Case | Model | Item | Before | After |
|---|---|---|---|---|---|
| held-out | val:32 | biobart-large-lora | c1 Patient is male | supported (span about the father) | supported (span: "sir") |
| held-out | val:36 | bart-base-lora | c2 Patient is female (she/her) | supported | unclear |
| held-out | val:51 | bart-base-lora | f1 Substance abuse | present (span: "you need it.") | present (span: "try to stop your substance abuse") |
| held-out | val:51 | biobart-large-lora | f1 Substance abuse | present (span: "you need it.") | unclear (span: "try to stop your substance abuse") |
| held-out | val:53 | biobart-large-lora | f9 Denies cough | omitted | unclear |
| held-out | val:83 | biobart-large-lora | c8 Patient is male (he) | (missing) | supported |
| held-out | val:94 | bart-base-lora | c1 Patient is male (his) | supported (span: "Mister Jay") | supported (span: "I reviewed his past medical history") |
| calibration | val:62 | biobart-large-lora | c17 Patient is female (she/her) | (missing) | supported |

## What moved

Rebuilding the published results from the revised labels changed these numbers. The headline rates not listed, including the patient demographic error rates (11% of BioBART-large summaries and 28% of BART-base summaries), are unchanged at the precision the README reports, though some confidence intervals shifted by a point.

| Result | Version 1 | Version 1.1 |
|---|---|---|
| BioBART-large + LoRA, expected facts omitted | 40 / 139 = 29% | 39 / 137 = 28% |
| BioBART-large + LoRA, labelled claims | 226 | 228 |
| BART-base + LoRA, labelled claims | 232 | 231 |
| Held-out facts the reviewer marks omitted | 46 | 45 |
| Judge omission precision and recall | 0.93 and 0.87 | 0.98 and 0.89 |
| Gender-reference rows in the detector table | sex-reference inconsistency (4 positives), sex contradiction (1), pronoun contradiction (0) | gender-reference contradiction (1 positive); the judge and the rule checker each found it |
| Judge-reviewer agreement | 0.992 raw over 1,746 claim-type pairs, kappa 0.88 | 0.995 raw over 1,293 pairs, no kappa |

The old "sex-reference inconsistency" type existed only because sex terms and pronouns were treated as separate dimensions. With one dimension it is the same thing as a gender-reference contradiction, so the claim types fell from seven to five. That is why the pair counts are smaller. Kappa is reported only when every cell of the agreement table has at least 5 pairs, and the merged type no longer meets that, so the file now reports raw agreement alone.
