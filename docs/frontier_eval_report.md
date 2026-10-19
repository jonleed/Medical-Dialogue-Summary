# Stronger clinical factuality evaluation — October 17–18, 2026

The published methods led to a stricter evaluation, and it exposes a real limitation: the existing demographic screen detects **36/51 contradictions (70.6%)** on the final broader synthetic challenge. Its previous **9/9 known-example result was a regression check**, not evidence of general accuracy. The screen fails the declared 90% engineering detection target.

The [research memo](frontier_eval_research.md) compares OpenAI, Anthropic, Google, FActScore, FENICE and medical error-injection research, with primary-source links and a mapping to the implementation. These are adaptations of public methods; no private frontier-lab infrastructure or official benchmark result is claimed.

## Actual measurements

Final suite: **130 unique synthetic cases, 38 template families, 56 correct/corrupted pairs**. This includes 114 demographic cases and 16 clinical scope probes. Labels are controlled construction labels, not independent clinician gold. Number substitutions within a family are correlated; intervals resample whole families. The original checker was frozen and hash-verified, and was not patched against these new examples.

| Full-source screening metric | Observed result |
|---|---:|
| Demographic contradictions detected | **36/51 = 70.6%** |
| Supported demographic properties falsely contradicted | **0/48** |
| Demographic property extraction coverage | **96/114 = 84.2%** |
| Exact four-label agreement on demographic construction labels | **74/114 = 64.9%** |
| Demographic matched pairs with both members correct | **33/48 = 68.8%** |
| Clinical-scope contradictions detected | **0/8** |

The demographic family-clustered 95% bootstrap interval for contradiction sensitivity is **47.1–92.7%**, describing variation across this designed suite rather than a clinical population. Zero observed false contradictions does not establish a zero false-positive rate: the binary any-false-alarm Wilson interval across 22 supported families extends to **14.9%**. The empirical bootstrap's zero-error interval is degenerate; it is explicitly flagged as insufficient to bound real-world risk.

The **clinical scope probes are outside the demographic screen's implemented scope**. Their zero detection result demonstrates why that screen cannot be the general clinical judge. Software tests passing does not imply these capability targets pass.

The separate model-visible evaluation detects **33/48 demographic contradictions**. Three full-source age contradictions become unsupported claims relative to the truncated visible evidence. Generated-property extraction and availability of a visible-source judgment are counted separately: no source assessment is not silently called a missing generated claim or a pass.

Source text and generated target spans, per-family results, confusion matrices, per-class precision/recall, macro F1, missing/unassessed outcomes and matched-pair results are in the machine artifacts. Common misses include `aged 46`, `46 y/o`, `Age: 46`, bare age answers, and family answers that establish the patient's pronoun. Missing-source demographic assertions usually become review candidates rather than definite unsupported labels.

## Benchmark quality audit and version history

An initial pilot contained **204 rows but only 130 distinct source/prediction/type/view records**. Independent code review also found that aspirin did not exclude metformin, diabetes did not exclude asthma, and a pronoun-only source could not support a compound pain assertion. These are benchmark faults, not model failures.

The pilot is preserved under `challenge/` and `screening-baseline/` and is **superseded**. The corrected versions remove exact duplicates, require explicit source exclusions for medication/diagnosis contradictions, and narrow demographic targets to the specific typed property. The final `challenge-final/` is revision v3: it also explicitly excludes the alternative identity rather than assuming sex/gender categories are inherently exclusive. Paired edits are programmatically reconstructed and evidence spans are checked. The corrected frozen cases were then rescored. `screening-final/` is authoritative; the intermediate `screening-baseline-v2/` predates the distinction between unassessed visible evidence and unextracted output properties. The earlier `challenge-v2/` and `screening-baseline-final/` are retained for provenance; their principal counts agree with the final revision. These source/gold corrections did not change the checker or improve its measured detection count.

This challenge was authored by the implementer familiar with the code, and its pilot was inspected for quality. **It is not a private holdout or an independent validation set.** Now that its failures are public, it is development/capability data. An independent clinically reviewed holdout remains necessary.

## Semantic judging and independent review workflow

The new provider-neutral workflow exports evidence-grounded tasks and validates actual returned judgments. It supports `supported`, `contradicted`, `unsupported` and `unclear`, positive source evidence for support/contradiction, exact view-specific spans, judge identity, pinned model revision, declared model family, settings and trial number. Repeated aliases of the same model cannot manufacture extra judges; settings are retained per trial. Missing generations are rejected rather than converted into completed empty inventories.

Two task types are separate:

- **Whole-summary inventories:** atomize all factual assertions in every saved summary, rather than starting with errors the checker already found. Completeness remains reviewer-attested and needs independent review. No automatic whole-summary pass is awarded.
- **Fixed-property calibration:** grade known properties without showing the expected label, author model, reference summary or checker result. Synthetic and existing agent references remain distinct; neither becomes human gold.

The completed export contains **964 distinct tasks**: 600 whole-summary inventory tasks (**300 saved summaries × 2 evidence views**) and 364 fixed-property tasks. There are **363 reference labels**: 260 synthetic labels and 103 eligible agent labels; one truncated real-case visible judgment has no reference label and remains unscored. **100 source-only expected-fact planning packets** are also prepared, all pending.

Reviewer A/B packets are split by evidence view and task stage. Independent visible-only reviewers must freeze their judgments before any full-source or source-planning exposure. Full-summary inventories must precede known-property calibration for a case, to avoid cueing the inventory. Model calls require fresh isolated task contexts. The root task index and private mapping/reference files are administrative artifacts, not a reviewer packet.

Actual new semantic-model judgments: **0**. Actual new human judgments: **0**. All **964 tasks remain pending**, no full-summary inventory is complete, and no semantic-judge accuracy or omission rate is claimed. Model agreement, once collected, will be a proposal for adjudication, not human gold. Two qualified independent clinicians, an adjudication process, and a clinically reviewed holdout are still needed to establish clinical evaluator validity.

## Files and execution

- [Research and source comparison](frontier_eval_research.md)
- [Design](frontier_eval_design.md) and [implementation plan](superpowers/plans/2026-10-17-frontier-evaluation.md)
- [Final frozen challenge manifest](../results/factuality/frontier-methods-2026-10-17/challenge-final/manifest.json)
- [Final screening metrics](../results/factuality/frontier-methods-2026-10-17/screening-final/metrics.json)
- [Every baseline decision](../results/factuality/frontier-methods-2026-10-17/screening-final/decisions.jsonl)
- [Pending judge report](../results/factuality/frontier-methods-2026-10-17/final_pending_judge_report.json)
- [Verification and qualification status](../results/factuality/frontier-methods-2026-10-17/validation.json)

New modules: `factuality/meta.py`, `factuality/challenge.py`, `factuality/judging.py`, and `factuality/prompts/claim_judge_v1.txt`; CLI stages added in `factuality/__main__.py`; meaningful invariant/regression tests in `tests/test_frontier_eval.py`. Real review tasks, mappings and reference labels remain ignored/local under `results/factuality/local/frontier-methods-2026-10-17/judges-final/`. Share only an appropriate view/stage task packet with its reviewer.

**87/87 offline software tests pass** (78 evaluator/pipeline tests and 9 existing app tests). Independent review findings were reproduced and fixed with tests. Training, serving, historical result files, saved predictions and previous demographic-checker results are preserved. No new test-split evaluation, model generation, paid API call or publication occurred.

## Reproduce

From the repository root, use fresh output names:

```bash
# Re-run the final frozen challenge against its unchanged screening baseline.
.venv/bin/python -m factuality challenge-run \
  --frozen results/factuality/frontier-methods-2026-10-17/challenge-final \
  --out results/factuality/local/frontier-baseline-rerun \
  --require-targets
# Expected: result files are written, then exit status 2 because detection < 90%.

# Export the 300 saved outputs, selected development claims and synthetic challenge.
.venv/bin/python -m factuality judge-export \
  --cases results/factuality/local/biobart-flask-2026-10-17/cases.jsonl \
          results/factuality/local/bart-lora-flask-2026-10-17/cases.jsonl \
          results/factuality/local/bart-base-flask-2026-10-17/cases.jsonl \
  --annotations results/factuality/local/demographics-review-2026-10-17/biobart_annotations.jsonl \
                results/factuality/local/demographics-review-2026-10-17/bart-lora_annotations.jsonl \
                results/factuality/local/demographics-review-2026-10-17/bart-base_annotations.jsonl \
  --challenge results/factuality/frontier-methods-2026-10-17/challenge-final \
  --out results/factuality/local/frontier-judge-rerun

# After actual independent judgments, validate/import files without overwriting them.
.venv/bin/python -m factuality judge-score \
  --tasks results/factuality/local/frontier-judge-rerun/tasks.jsonl \
  --responses /path/to/actual-judge-A-responses.jsonl /path/to/actual-judge-B-responses.jsonl \
  --reference-labels results/factuality/local/frontier-judge-rerun/private_reference_labels.jsonl \
  --out results/factuality/local/frontier-judge-rerun/actual-judge-report.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
```

`judge-export` makes no model calls. Run each authorized semantic judge on the supplied isolated task JSON and return its actual identity/settings and responses. The fixed-claim response shape is:

```json
{
  "task_id": "copy from task", "task_hash": "copy from task", "trial": 0,
  "reviewer": {"kind": "model", "id": "actual judge ID", "model_revision": "actual pinned revision", "model_family": "actual family", "parameters": {}},
  "label": "unclear", "rationale": "Actual evidence-based rationale", "evidence": []
}
```

For a whole-summary inventory, replace the top-level label/rationale/evidence with `claims_complete` and a `claims` list; every claim contains label/rationale/evidence plus `generated_span` and `fact_type`. These are response-format instructions, not recorded model judgments. Existing canonical source-only planning/annotation commands remain available for a real expert omission study; the new planning packets are pending aids, not frozen omission gold.
