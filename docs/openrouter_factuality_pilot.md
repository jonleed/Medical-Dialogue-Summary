# Free API targeted factuality evaluation — October 18, 2026

## Reviewer choice

Use **`nvidia/nemotron-3-ultra-550b-a55b:free`**, pinned to the NVIDIA endpoint through OpenRouter. The live model and endpoint catalogs list zero prompt and completion prices. This is the selected practical reviewer, not a demonstrated winner on clinical factuality.

The [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) reports instruction-following, long-context and non-hallucination evaluations. Those make it a reasonable choice for comparing claims against supplied evidence. They do not measure this evaluator's age/gender contradictions, unsupported claims or omission accuracy. The card's OmniScience non-hallucination result also comes with lower answer accuracy; abstention behavior must not be mistaken for complete detection. Published benchmark settings differ from this short, reasoning-disabled JSON review.

The current free catalog was inspected for other candidates, including Inkling, Gemma 4, Dots3-Note Preview and smaller Nemotron variants. No inference comparison or model ranking was run. Broad reasoning scores alone do not establish the best clinical reviewer. [Inkling's card](https://thinkingmachines.ai/model-card/inkling/) and the [Dots3 model repository](https://github.com/studio-dots-ai/dots3-note-prev) describe general models, without direct validation for these four endpoints. [Live OpenRouter catalog](https://openrouter.ai/api/v1/models).

## Execution boundaries

The adapter reads the key locally without shell sourcing or displaying it. It accepts only the selected model and provider, verifies zero endpoint prices before dispatch, and sends server-side zero maximum prices with provider fallback disabled. It exposes no tools or plugins and makes one API completion per case. It rejects model/provider substitutions, nonzero or unknown actual cost, incomplete output, invalid plan IDs and fabricated quotations. [OpenRouter routing controls](https://openrouter.ai/docs/guides/routing/provider-selection).

The original 24-attempt cap includes all five failed OpenCode attempts. The OpenRouter ledger retains those reservations and permits at most **19 new requests**. The historical OpenCode ledger remains blocked. There are no extra Codex inference workers, model comparisons, paid fallbacks or automatic retries.

## Development packet and prompt correction

All 24 constructed cases and source-only expected-fact plans were frozen before the free-model run. A predeclared 19-case subset covers four age cases, six sex/gender/pronoun cases, three unsupported-claim cases and six omission/exclusion cases. Five cases remain outside the dispatch budget and stay in the full-packet coverage denominator. These are implementer-authored development fixtures, not an independent clinical test set.

The first API response identified its age contradiction correctly but invented an expected-fact ID even though its supplied plan was empty. The strict validator rejected the complete review. That failure remains recorded and was not retried. Prompt revision 2 explicitly confines `expected_facts` to entries in `plans`; it applies only to unattempted development cases. The manifest retains both prompt revisions. This is development iteration, not held-out evaluator calibration.

## Reproduction

```bash
# Offline tests: no inference.
.venv/bin/python -m unittest discover -s tests -v

# Offline report from saved reviews: no inference.
.venv/bin/python -m factuality.openrouter_pilot report
```

Raw requests, responses, key-free preflight metadata, source cases, manifest, ledger and local review HTML stay under ignored `results/factuality/local/openrouter-pilot-2026-10-18/`. Public aggregates are in `results/factuality/openrouter-pilot-2026-10-18/`.

Human review, independent annotation agreement, reviewer sensitivity on natural clinical mistakes and factuality assessment of all 300 saved summaries remain outstanding. A successful development packet does not establish frontier-equivalent or clinician-equivalent accuracy. The actual served weight revision and quantization are not exposed by the free endpoint.

## Actual results

| Capability | Planned requests | Valid reviews | Correct target decisions |
|---|---:|---:|---:|
| Patient age / relative-age controls | 4 | 3 | 2 |
| Sex, gender identity and pronouns | 6 | 6 | 5 |
| Unsupported claims | 3 | 3 | 2 |
| Omissions / optional and section exclusions | 6 | 5 | 5 |
| Total | **19** | **17** | **14** |

All 19 OpenRouter responses reported **$0 actual cost**, totaling 12,550 input and 2,640 output tokens. The five historical OpenCode attempts still have unknown actual usage and retain full reservations. No additional inference capacity remains in the 24-attempt pilot cap.

Coverage is **17/19 valid reviews** on the dispatched subset and **17/24** on the full frozen packet. Target extraction is **15/24** and end-to-end correct target decisions are **14/24** when all frozen cases remain in the denominator. Five frozen cases were not dispatched. None of these are clinical accuracy estimates.

Observed gaps:

- Case 01: the initial prompt produced an invented expected-fact ID; the entire review was rejected. Its contradiction classification was correct, but is not counted as a valid review.
- Case 06: a father's age was labeled `age` despite the rubric assigning relative ages to `quantity`; the required target type was not extracted.
- Case 12: the brother's cough was reviewed as a symptom, with no separate pronoun target extracted. The supported symptom does not establish pronoun-coverage completeness.
- Case 14: absence of medication evidence was labeled `contradicted` instead of `unsupported`. The rationale itself said metformin was not mentioned.
- Case 21: a source-only phrase, “Return tomorrow,” was offered as evidence that the summary contained the follow-up time. The validator rejected it because that phrase is absent from the summary. This is a substantive omission-review failure, not just JSON formatting.

The prompt was not tuned after these latter findings, and failed cases were not retried. Evidence-invalid reviews remain failed assessments while processing other predeclared cases; provider, route, cost and token failures still stop execution persistently. Optional/out-of-section labels are enforced by local source-plan eligibility and original model proposals are preserved. Their successful end-to-end decisions are not solely model ability.

The project can truthfully describe a **targeted factuality evaluation for age and gender contradictions, unsupported claims, and omissions in clinical summaries**, with explicit coverage and validation failures. It cannot yet claim independent clinical validation or frontier-level evaluator accuracy. Further work should address these observed failure modes and evaluate an independent, clinician-annotated sample.
