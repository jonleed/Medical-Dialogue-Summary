# Targeted factuality evaluation

The framework evaluates age and gender contradictions, unsupported claims and requested-section omissions. It applies published frontier-lab evaluation methods with deterministic local checks and a pinned free semantic reviewer. Training and serving behavior are unchanged by this evaluation.

## Design

- Separate full-source fidelity from fidelity to the input actually visible to the summarizer. Token truncation does not become an unexplained model contradiction.
- Resolve patient versus relative attribution, and keep sex, gender identity and pronouns separate. Honorifics alone do not establish these attributes. The local screen does not implement race/ethnicity evaluation.
- Freeze cases and source-only expected-fact plans. Do not derive omissions from an unseen reference note or invent expected facts after examining the output.
- Blind reviewer payloads to construction labels and author-model identity. Keep deterministic proposals separate from final judgments; the free pilot deliberately includes local rule proposals as reviewer aids.
- Require exact source evidence for support/contradiction and exact summary evidence for an asserted present fact. Missing source evidence means unsupported or unclear, not automatically contradicted. An absence judgment still depends on reviewing the full source.
- Retain invalid reviews, missed extractions, exclusions and unattempted targets in explicit coverage reporting. Optional/out-of-section facts do not become omissions.
- Record reviewer identity, settings, frozen request identities, actual token usage and cost. Keep model suggestions separate from clinician annotations.

[Published methods and adaptations](frontier_eval_research.md) · [Annotation protocol](factuality_protocol.md) · [JSONL schema](factuality_schema.md).

## What was executed

### Local demographic screening

The checker processed 300 saved summaries from 100 validation dialogues across three summarizer configurations. The [aggregate](../results/factuality/saved-summary-screening-2026-10-18/screening_results.json) contains per-model counts and model-visible contradiction flags. These are automatic suggestions with unestablished precision/recall, not annotated clinical error rates or a factuality ranking.

The BioBART run has 35 unclear suggestions across 26 summaries; the BART LoRA run has 54 across 38. Some summaries contain multiple suggestions. This does not establish a training-data cause. BART-base's two contradiction suggestions occur in one summary; fewer flags do not prove better summaries.

### Free semantic development pilot

The [pilot report](openrouter_factuality_pilot.md) documents the 24 constructed cases, 19 attempted cases, 17 valid reviews, 14 correct target decisions and retained failures. All 19 OpenRouter responses reported zero cost. Full-packet coverage remains separate from performance on assessed targets. A prompt revision after the initial schema failure is disclosed; this is development iteration, not an untouched clinical holdout.

Unsupported medication/diagnosis claims and section omissions have not been semantically evaluated across all 300 saved summaries. No clinician annotations or clinical accuracy rates were produced in these runs.

## Input and scoring provenance

The saved summaries use the Flask note route with 512-token inputs, deterministic four-beam decoding, repetition penalty 2.0, no-repeat four-grams and output max_length 512. Private per-run manifests retain case IDs/hashes, input IDs/masks, model/adapter identity and generation settings. They remain under ignored `results/factuality/local/`.

The current 100-dialogue cohort is validation data already used for checkpoint selection. It supports development analysis; it is not a final held-out test. Historical cached integration predictions without adequate source/model/input identity are not relabeled as this run.

ROUGE is reported separately. The notebook uses a 128-token generation limit and differs from Flask in section mapping, so notebook and Flask scores should not be pooled. Both protocols are preserved in `eval_notebook_protocol.py` and `app.py`.

## Reproduction

Run offline checks from the repository root after installing requirements:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
python -m factuality --help
```

To export new deterministic validation cases and generate through the Flask route, use fresh output paths:

```bash
mkdir -p results/factuality/local
python -m factuality export --n 100 --seed 42 \
  --out results/factuality/local/repro-validation.jsonl
python -m factuality generate \
  --cases results/factuality/local/repro-validation.jsonl \
  --out results/factuality/local/repro-biobart \
  --base-model GanjinZero/biobart-large --adapter ./biobart-large-lora
```

Generation needs the model weights and is separate from the offline test suite. The batch screening command expects the existing named local runs; it does not regenerate them. Saved pilot reporting likewise requires its local manifest, frozen cases, ledger and review responses. Public aggregates alone do not recreate raw predictions or judge responses.

The API pilot pins a free NVIDIA endpoint, checks prices and sends zero maximum prices with fallback disabled. Its exhausted attempt ledger prevents further inference. Credentials are read locally and never included in requests saved for review. See the pilot report before any explicitly scoped future run.
