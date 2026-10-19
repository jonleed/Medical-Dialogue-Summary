# Factuality pilot (October 2026)

Before building a labeled benchmark, a small pilot checked two things: what a rule-based screen finds in saved summaries, and whether a free LLM can act as an evidence-grounded semantic reviewer. Everything here is development-level evidence. None of it is an accuracy estimate, and none of it is clinical validation.

## Design

- **Visible input versus full source.** Fidelity to the full dialogue is kept separate from fidelity to the 512 tokens the summarizer could see, so truncation never becomes an unexplained contradiction.
- **Patient versus relative.** Ages, sex, gender identity and pronouns are attributed to the right person. Honorifics alone don't establish sex or pronouns.
- **Frozen inputs.** Cases and source-only expected-fact plans are fixed before any model output is read, so omissions aren't derived from the reference note or invented after seeing the summary.
- **Blinded, evidence-checked review.** The reviewer never sees construction labels or the source model's identity. Every supported or contradicted claim must quote the source, every asserted present fact must quote the summary, and the quotes are validated character by character.
- **Explicit coverage.** Invalid reviews, missed extractions, exclusions and unattempted cases stay in the denominators.
- **Recorded provenance.** The reviewer model, settings, request identities, token usage and cost are saved with the results.

## What ran

### Rule-based screening of saved summaries

A rule checker processed 300 saved validation summaries (100 dialogues across three summarizer configurations) for patient age, sex, gender identity and pronouns.

| Model | Summaries | Contradiction suggestions | Supported suggestions | Unclear suggestions | Summaries flagged for review |
|---|---:|---:|---:|---:|---:|
| BioBART-large with LoRA | 100 | 3 | 20 | 35 | 27 |
| BART-base with LoRA | 100 | 4 | 17 | 54 | 38 |
| BART-base, no adapter | 100 | 2 | 49 | 0 | 1 |

These are rule-generated suggestions, not verified errors. A summary can produce several, and the two BART-base contradictions come from one summary. Fewer flags don't mean better summaries: the base model's short outputs mention few demographics at all. The checker's precision and recall were not measured on this sample.

### Semantic review of constructed cases

The 24 constructed cases cover four capabilities, six each: age, sex/gender/pronouns, unsupported claims and omissions, each with three errors and three controls. Run locally, the rule checker flagged all 6 known age and gender contradictions and none of the 6 controls.

The semantic reviewer was `nvidia/nemotron-3-ultra-550b-a55b:free`, pinned through OpenRouter, with zero-price routing, no tools and one request per case. The 24-attempt budget was set up front. It also covered five failed attempts through an earlier free-model transport, which the provider blocked with a free-tier access check, so that route produced no reviews. That left 19 requests, all at zero actual cost (12,550 input and 2,640 output tokens). Five cases were not dispatched and stay in the full-packet denominator.

| Capability | Requests | Valid reviews | Correct target decisions |
|---|---:|---:|---:|
| Patient age / relative-age controls | 4 | 3 | 2 |
| Sex, gender identity and pronouns | 6 | 6 | 5 |
| Unsupported claims | 3 | 3 | 2 |
| Omissions / section exclusions | 6 | 5 | 5 |
| **Total** | **19** | **17** | **14** |

Across all 24 frozen cases that is 17 valid reviews, 15 correct target extractions and 14 correct end-to-end decisions.

The first response invented an expected-fact ID that its plan didn't contain, and the strict validator rejected the whole review. The prompt was revised once to confine expected facts to the plan, and only cases not yet attempted used the revision. The failed case was not retried.

## Failure modes

- A review invented a plan ID, so the entire review was rejected.
- A father's age was labeled as the patient's age instead of a relative quantity.
- A brother's cough was reviewed as a symptom, and no separate pronoun target was extracted.
- A medication that was never mentioned was labeled contradicted instead of unsupported.
- A source-only phrase ("Return tomorrow") was offered as evidence that the summary contained the follow-up, and the validator rejected it because the phrase isn't in the summary.

## Limits

- The 24 cases are constructed development fixtures, not an independent test set, and their failures are now public.
- The 300 summaries come from validation dialogues already used to select the summarizer checkpoint.
- Unsupported claims and omissions were assessed semantically only on the 24 constructed cases, not on the 300 saved summaries.
- A free endpoint doesn't expose the weight revision or quantization it serves.

## Input and scoring provenance

The saved summaries come from the Flask note route: 512-token inputs, deterministic four-beam decoding, repetition penalty 2.0, no-repeat four-grams and a 512-token output limit. Per-run manifests keep case IDs and hashes, input IDs and masks, model and adapter identity and generation settings, and they stay local.

The 100-dialogue cohort is validation data already used for checkpoint selection, so it supports development analysis and is not a final held-out test. Earlier cached integration predictions lack enough source, model and input identity, so they weren't relabeled as part of this run.

ROUGE is reported separately. The notebook protocol uses a 128-token generation limit and maps sections differently from the Flask route, so notebook and Flask scores shouldn't be pooled. Both protocols are preserved in `eval_notebook_protocol.py` and `app.py`.

## What came next

The pilot showed whether the reviewer followed its instructions and where the checker was unsure, but not whether either was right. Answering that needs reference labels for real summaries. The benchmark that follows uses plans and labels drafted by Claude Sonnet in an isolated session, with every item checked by a human reviewer.

## Reproduction

```bash
# Offline tests: no model calls.
python -m pytest -q

# Rebuild the screening aggregate when the saved local cases are available.
python -m factuality.evaluate_saved_summaries

# Rebuild the pilot report from saved reviews: no model calls.
python -m factuality.openrouter_pilot report
```

Aggregates are committed under `results/factuality/`. Raw dialogues, summaries, prompts and responses stay local.
