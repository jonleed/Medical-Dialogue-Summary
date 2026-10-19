# JSONL schema and review workflow

Run commands from the repository root. JSONL is UTF-8, one object per line. Version 1 is checked by `factuality.scoring.validate_annotation`; spans are end-exclusive Unicode character offsets, not bytes or token positions. Machine metrics encode N/A as JSON `null`.

## Case (`cases.jsonl`)

`schema_version`, `dataset_kind` (`real`/`synthetic`), `case_id`, `source` (split, stable dialogue ID, CSV/hash, original row indices, original headers, dialogue hash), `dialogue`, `requested_section`, `notebook_section`, `reference` (full notebook-mapped target), `reference_sections`, `prompt`, `protocol`, `generation_status`, `prediction`, `model_input`, `provenance`.

Before generation `prediction` and `model_input` are null. After generation, `model_input` contains `input_ids`, `attention_mask`, `text` (decoded nonpadding input), `full_token_count`, `truncated`, and token-ID `sha256`. `provenance` points to the run manifest and records base revision, adapter weight hash, actual generation time and duration. Manifest records code/data revisions, changed code hashes, tokenizer backend hash, adapter file hashes, full generation config, package versions, device, thread count, exact command and UTC timestamps. No access token is saved.

Raw artifacts stay under ignored `results/factuality/local/`. Copying their manifests/metrics to a tracked directory is deliberate; raw text and queues remain local.

## Source-only expected-fact plan (`plan`)

The blind plan omits predictions and model-visible text. Its `source_plan_hash` binds dialogue, source, section, and reference independently of whether generation has occurred. Review source and requested section, using reference only for intended coverage. Add expected and explicitly excluded facts **before** opening the prediction. Set `expected_facts_frozen`, `expected_facts_blind`, `expected_facts_reviewer`, `expected_facts_reviewer_kind`, and `expected_facts_frozen_at` honestly. `expected_facts_complete` means the source inventory is exhaustive for the requested section. A plan fact starts `pending` (or `excluded`); output-specific `present`/`omitted` labels and generated/visible evidence are filled during claim review, after the plan is frozen. The CLI cannot verify whether a person looked at a prediction; blindness is an explicit reviewer attestation.

If you already saw an output, do not attest blindness. Keep that case's omissions unscored or assign an independent reviewer who has not seen it. The provided expected-fact plans are empty and pending, not fabricated source inventories.

## Annotation (`annotations.jsonl`)

Each record has `case_id`, `case_hash` from `factuality.io.case_hash(case)`, `reviewer`, `reviewer_kind` (`human`, `agent`, `model`, `synthetic`), `status` (`pending`, `partial`, `complete`), `claims_complete`, `expected_facts_complete`, the freeze/attestation and reviewer-kind fields above, `claims`, `expected_facts`, `unresolved`.

Pending queue records have an empty reviewer, no labels, and an unresolved review task. Partial/final records require a real identity or a clearly named agent/model. Expected-fact reviewer kind must match the scoring stratum; imported agent/model plans retain their kind and cannot become human omissions automatically. One record per case per reviewer kind is accepted; resolve multiple reviews explicitly rather than silently counting both. This format does not calculate agreement.

Each atomic generated claim has:

```json
{
  "id": "c1", "text": "The patient reports pain", "fact_type": "symptom",
  "label": "supported", "visibility": "visible",
  "generated_span": {"start": 0, "end": 24, "text": "The patient reports pain"},
  "source_spans": [{"start": 9, "end": 20, "text": "I have pain"}],
  "visible_spans": [{"start": 9, "end": 20, "text": "I have pain"}],
  "rationale": "Patient explicitly reports pain."
}
```

Offsets in this example apply to synthetic strings `The patient reports pain` and `Patient: I have pain`. Fact types and labels are listed in the protocol. Expected-fact objects use the same ID/text/type/visibility/evidence/rationale fields, plus labels `present`, `omitted`, `unclear`, `pending`, `excluded`. Only `present` requires a `generated_span`. Excluded facts need a section-relevance rationale. Unsupported claims need a rationale explaining full-source inspection; no positive evidence span can prove absence. Supported/contradicted claims and relevant facts require positive original evidence; model-visible assessability additionally requires visible evidence.

`status=complete` requires both completeness booleans. Pending/unclear/unresolved items still prevent a passing summary. Annotation completeness is reviewer-attested; software cannot establish exhaustive atomization or clinical relevance. All spans and hashes are checked; malformed or stale annotations fail rather than quietly disappearing from a denominator.

## Automatic suggestions and scores

`automatic_suggestions.jsonl` is separate from annotation JSONL. Current method `patient_demographics_v2` emits `generated_span`, `source_spans`, `visible_spans`, `fact_type`, `proposed_label`, `model_visible_proposed_label`, `visibility`, `generated_value`, `source_values`, `issue`, `evidence_bases`, `review_required`, rationale and `status=pending`. The `check` export additionally binds each suggestion to `case_hash`. Every span uses exact character offsets in its corresponding text.

The v2 checker handles spelled-out/current patient ages, age question/answer evidence, patient-attributed pronouns and explicit same-dimension identity statements. Honorific mismatches and missing/conflicting/qualified evidence yield `unclear` review candidates. No automatic suggestion becomes a human label, a final unsupported judgment, or an omission. `model_visible_proposed_label=null` means no reliable same-dimension patient evidence was extracted from the visible input. Full-source evidence lost to truncation cannot establish a model-visible contradiction. See [v2 results and limits](factuality_report.md).

The automatic score block includes `checker_version`, supported/contradiction/unclear suggestion counts, `flagged_cases` (any review-required suggestion), `contradiction_flagged_cases`, `model_visible_contradictions`, and `truncated_source_contradictions`. Earlier saved v1 metrics had only contradiction-flagged cases and a much narrower extractor. Historical snapshots are not rewritten. Checker precision/recall remain null without independent validation.

`python -m factuality check --cases <cases.jsonl> --out <new-directory>` writes automatic suggestions and a hash-bound audit summary. Optional `--annotations <reviewed.jsonl>` compares selected existing labels for development regressions; this does not create an independent validation set or populate annotations. Source and annotation hashes and checker-code hashes identify exactly what was checked.

`score` produces separate `human`, `agent`, `model`, `synthetic` strata, each with `full_source` and `model_visible` metrics. Each metric has numerator, eligible denominator and rate/null. Counts include complete/partial/pending cases, unclear items, excluded facts, and truncation coverage loss. Sparse agent example annotations are selected illustrations, not empirical population estimates.

The CSV and HTML review views are read-only aids. Edit the canonical annotation JSONL with a text editor or script. Use `factuality.io.case_hash` for identity and the following helper for exact spans:

```python
def evidence(text, phrase):
    start = text.index(phrase)  # select the intended occurrence if repeated
    return {"start": start, "end": start + len(phrase), "text": phrase}
```

The general annotation commands do not call a judge. The separate free-model pilot records model identity, prompt, settings and responses as model review, never human gold; see [the pilot report](openrouter_factuality_pilot.md).

## Frozen capability tests and semantic judge tasks

See `python -m factuality challenge-freeze --help`, `challenge-run --help`, `judge-export --help` and `judge-score --help`. The [execution report](factuality_report.md) documents exact formats, blinding order, pending counts and commands. Judge task hashes bind the source view, prediction/target and rubric; imported positive judgments need exact evidence spans in that view. Distinct trials retain distinct configs, duplicate model aliases cannot inflate consensus, and ungraded/unextracted/unassessed items remain in their appropriate denominators. Summary inventories are separate from fixed-property calibration and require an honest `claims_complete` attestation; no whole-summary pass or omission score is produced from them automatically.
