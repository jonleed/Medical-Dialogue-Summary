# Judge rubric calibration

The judge rubric (`factuality/prompts/judge.txt`) was calibrated on the 30 calibration summaries: 15 dialogues, both summarizers. These summaries are never used for held-out results. Calibration ran in two rounds, and the rubric was frozen after the second.

## What the first round showed

Compared with the reference labels:

- **Unsupported claims (recall 0.31, precision 0.42).** The judge labelled 5 unsupported claims as contradicted because the dialogue described a *different* event (for example "punched a glass" against a summary saying "punched another guy"). A statement about a different event leaves the claim unsupported, not contradicted.
- **Claim granularity.** The judge often bundled several facts that the reference labels list separately. Claims are matched one to one, so about 24 reference claims went unmatched.
- **Quotes.** One review failed validation because a quote spanned two speaker turns and changed the original spacing.

## What the rubric clarifies

These clarify rules the annotation protocol already states. They aren't answers to specific cases.

1. **Contradicted** requires incompatible evidence about the same event, attribute or person. If the dialogue never addresses the claimed event, the claim is unsupported.
2. **One claim, one fact.** Each symptom, body site or side, duration or timing, medication, dose or frequency, negation and relationship is its own claim, with its own short quote.
3. **Quotes** are one contiguous passage, copied exactly, ideally within one speaker turn. A question and its answer are separate quotes, and the expected-fact inventory quotes the answer's own turn.

## Effect on calibration

| Measure | First round | Frozen rubric |
|---|---:|---:|
| Reference claims the judge didn't match | 45 | 32 |
| Unsupported claims, precision / recall | 0.42 / 0.31 | 0.55 / 0.38 |
| Omissions, precision / recall | 0.93 / 0.74 | 0.90 / 0.81 |
| Pronoun contradictions found | 0 of 2 | 1 of 2 |
| Sex contradictions found | 3 of 3 | 2 of 3 |
| Age contradictions found | 6 of 6 | 6 of 6 |

One review still fails quote validation, for a quote that runs into the next speaker turn, and is recorded as a failed assessment. The rubric was frozen at this point: further rounds would fit the rubric to these 30 summaries rather than clarify rules.
