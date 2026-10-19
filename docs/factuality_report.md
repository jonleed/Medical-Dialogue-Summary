# Exploratory factuality evaluation — October 17, 2026

The current BioBART-large + LoRA model generated **100/100 validation summaries** locally through `/generate_note`. **Zero cases have human factuality review; all 100 remain pending.** Human age/gender contradiction, unsupported-claim, omission and summary-error rates are therefore **N/A (0 eligible annotated items/cases)**. No pending case is counted as passing. The engineering rubric is written; human pilot count is zero and a human annotation freeze has not occurred.

See the [protocol](factuality_protocol.md), [JSONL schema and review instructions](factuality_schema.md), [run manifest](../results/factuality/biobart-flask-2026-10-17/run_manifest.json), [machine-readable metrics](../results/factuality/biobart-flask-2026-10-17/metrics.json), and [selected inspected examples](../results/factuality/inspected_examples.json). Raw source/prediction exports, source-only plans, annotation templates, CSV and HTML queues remain local under ignored `results/factuality/local/`.

A subsequent [focused age and gender-reference review](demographics_review.md) adds agent-checked demographic findings and describes the original checker’s missed cases. Its annotations are a separate review phase; the initial run artifacts below are preserved. The later [demographic checker v2 report](demographic_checker_v2_report.md) documents the automatic fix and a separate rerun of all 300 saved outputs. Counts and tests below describe the original evaluation phase.

## Completed generation and separate overlap results

All three configurations completed **100/100 cases each (300 summaries, 100 unique source dialogues)**. Human-reviewed counts are **0/300**, pending **300/300**; no complete real-case factuality reviews or omission inventories exist. There were no external model-judge runs. Eight current-model cases have selected agent annotations; all other claims remain unannotated.

| Model, Flask protocol | Summaries | ROUGE-1 | ROUGE-2 | ROUGE-L | Human factuality status |
|---|---:|---:|---:|---:|---|
| BART-base, no adapter | 100 | 0.2386 | 0.0828 | 0.1778 | 100 pending |
| BART-base + LoRA | 100 | 0.5055 | 0.2886 | 0.4378 | 100 pending |
| BioBART-large + LoRA | 100 | 0.5520 | 0.3259 | 0.4726 | 100 pending |

These are arithmetic mean F1 scores with stemming, full untruncated notebook-mapped references, and existing newline boundaries for ROUGE-Lsum. ROUGE-Lsum equals ROUGE-L here. Target truncation and aggregation differ from historical evaluations, so this table is a separate run. It does **not** rank factuality or safety. All three runs have the same three truncated inputs.

The [paired comparison](../results/factuality/paired_comparison.json) contains 100 per-case rows and verifies identical case order, prompts, source hashes, input IDs and decoding settings. Each model has separate manifests, metrics and pending queue summaries under `results/factuality/`. [Execution commands and counts](../results/factuality/execution_record.json) and [test/file-preservation evidence](../results/factuality/validation.json) are saved. **46/46 tests passed**: 37 offline evaluator/pipeline tests plus 9 existing app tests. The [known-label synthetic set](../results/factuality/synthetic_metrics.json) contains **17 hand-authored cases**, separate from real outputs; these fixtures require no model inference. Four independent code-review findings were reproduced and fixed with regression tests.

## Observations supported by inspected outputs

These are eight selected, **partially agent-annotated** real cases: 15 atomic propositions (7 supported, 7 contradicted, 1 unsupported). They are not a random reviewed sample, human gold labels, or empirical population error-rate estimates. Their omission inventories are unreviewed; no omission claims are scored. Source and generated spans, original CSV/dialogue hashes and whole-case hashes are in the example artifact. All illustrated evidence was visible to the model.

| Case | Original dialogue evidence | Selected generated claim | Agent inspection |
|---|---|---|---|
| `val:84` | “diarrhea and pain in my stomach”; “I threw up this morning” | “Abdominal pain, diarrhea, vomiting.” | Three supported symptoms |
| `val:57` | Asked about medications, patient answers “No”; OTC drugs: “None” | “medications: None.” | Supported absence |
| `val:72` | “high blood pressure”; “depression”; “osteoporosis” | “depression, hypertension, and osteoporosis” | Three supported conditions |
| `val:5` | “He had jaundice but they didn't give any treatment for it.” | “he had no jaundice” | Contradicted negation |
| `val:0` | “January tenth two thousand nine”; follow-up “February tenth two thousand nine” | “02/02/2009”; follow-up “02/03/2009” | Two contradicted dates |
| `val:36` | “abnormal levels for Sodium and Creatinine”; “thyroid levels are not normal” | “normal creatinine, normal sodium, normal thyroid” | Three contradicted lab assertions |
| `val:47` | Flagyl “five hundred M G every eight hours” | “Flagyl 5 mg every 8 hours” | Contradicted quantity |
| `val:41` | Urinary frequency and planned urine testing for infection; no stone diagnosis stated | “Urolithiasis” | Unsupported diagnosis after full-source and visible-input inspection |

MTS-Dialog: Ben Abacha et al., EACL 2023, [source repository](https://github.com/abachaa/MTS-Dialog), CC BY 4.0; [license](../MTS-Dialog/LICENSE.txt). Excerpts are selected from concatenated/trimmed dialogues; predictions and agent annotations are additions. Original licensing and disclaimers apply.

These examples show the distinction between overlap and factual evidence. `val:5` has ROUGE-L **0.5507** despite reversing jaundice; the correctly rendered medication-absence case has ROUGE-L **1.0000**. The Flagyl dose-error case still has ROUGE-L **0.7097** despite changing 500 mg to 5 mg. The date-error case has ROUGE-L **0.4189**; the lab-error case **0.1333**. ROUGE can decrease when facts change, but it does not identify which assertions are contradicted or distinguish negation and attribution reliably. These examples do not establish a correlation, error prevalence or clinical safety.

## Coverage and protocol audit

- Sampling exactly matches the historical grouped-ID validation preprocessing and pandas seed-42 ordering. All 100 grouped validation dialogues are included; this split already participated in model selection.
- **3/100 inputs were truncated**: `val:37` (850 tokens), `val:14` (623), `val:9` (527). Of 15,519 full input tokens, 15,055 were visible; **464 input tokens were discarded**. This is token coverage, not relevant-fact coverage. Truncation-related expected-fact loss is **N/A**, because no blinded expected-fact inventory exists yet.
- The intentionally narrow demographic checker made **0 suggestions**, including **0 contradiction flags**, on the current model's 100 real cases. It abstains on spelled-out ages, question/answer resolution, indirect identity and general clinical assertions. This is not evidence of demographic accuracy. Checker precision/recall on real cases are N/A.
- Notebook output max_length is 128; Flask is 512. `EXAM` becomes `exam` in training/notebook and `physical exam` in Flask. There are six actual section-mapping differences: one `EXAM` and five unknown headers (`GYNHX`, `DIAGNOSIS`, `IMAGING`, `IMMUNIZATIONS`, `OTHER_HISTORY`) that yield an empty notebook section but Flask HPI fallback. `EDCOURSE` and `FAM/SOCHX` labels are retained as implemented. See the [mapping audit](../results/factuality/protocol_audit.json). These behaviors were documented, not changed.
- Historical cached integration predictions are [unscorable for this evaluation](../results/factuality/cache_audit.json): references alone cannot prove source IDs, model identity, exact input or generation provenance. README's historical integration score and the saved cache's score are also different; neither is relabeled as the new run.
- The generation harness suppresses only the test-data UI sample-table initialization on import. It does not read the test CSV or evaluate test examples. Existing regression tests still exercise the unchanged application.

## Execution and reproducibility

The run began October 17, 2026 in America/New_York (October 18 UTC). Code started from Git revision `0085700`; exact full revision, dataset submodule revision, package versions, timestamps, command and artifact hashes are recorded in each manifest. The dataset is at `3ff0801933608d6f570468c13125125fb5cabdea`. BioBART base revision is `47c968d3aba260d5737fda6f5b2ad3aa35ce75e8`; selected adapter weights SHA256 is `eb107774f60fc3a80ea8855827ef98dda3cca1b434ae8a1e8cae5f8f22f556e5`.

Execution used the existing Python 3.14.8 environment, PyTorch 2.13.0, Transformers 5.15.1, PEFT 0.20.0 and CPU. BioBART used four PyTorch threads; BART comparisons use two. Decoding is deterministic, four beams, repetition penalty 2.0, no-repeat four-grams, 512-token padded/truncated inputs and output max_length 512. Input IDs and attention masks are saved per case. [Pinned model generation defaults](../results/factuality/model_defaults.json) were separately reconstructed from their cached configs; this is identified as a post-run audit, not a runtime capture. No training, serving modification, external judge, paid API, push or publication occurred.

Generation started while the annotation/reporting harness was being completed. [Exact generation-stage module snapshots](../results/factuality/generation_code_snapshots.json) match the hashes captured at process startup. Later request/provenance checks were applied retrospectively to saved cases as well as added to the final harness; they do not change model decoding. All historical result files remain unchanged.

Run from the repository root. Use fresh output names: export/generation/queue stages intentionally refuse to overwrite existing work.

```bash
# Offline diagnostics and existing app regressions; no model weight download.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q

mkdir -p results/factuality/local
.venv/bin/python -m factuality export --n 100 --seed 42 \
  --out results/factuality/local/repro-validation.jsonl
.venv/bin/python -m factuality plan \
  --cases results/factuality/local/repro-validation.jsonl \
  --out results/factuality/local/repro-source-only.jsonl

# Pin to the actual cached revision. For a fresh machine, fetch this exact
# revision using snapshot_download(..., local_files_only=False) first.
FACTUALITY_BIOBART_PATH=$(.venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("GanjinZero/biobart-large", revision="47c968d3aba260d5737fda6f5b2ad3aa35ce75e8", local_files_only=True))')
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m factuality generate \
  --cases results/factuality/local/repro-validation.jsonl \
  --out results/factuality/local/repro-biobart --threads 4 \
  --base-model "$FACTUALITY_BIOBART_PATH" --adapter ./biobart-large-lora
.venv/bin/python -m factuality summarize \
  --run results/factuality/local/repro-biobart \
  --out results/factuality/repro-biobart
```

Summarization writes hashed sample metadata, machine metrics, separate ROUGE and a local pending queue. Default summarization gives zero human-reviewed cases. For source-only planning use the blind file first. After genuinely completing a pilot and freezing the rubric, fill/attest the source-only plans before viewing predictions; import them with:

```bash
.venv/bin/python -m factuality queue \
  --cases results/factuality/local/repro-biobart/cases.jsonl \
  --plans results/factuality/local/repro-source-only.jsonl \
  --out results/factuality/local/repro-planned-review
# Edit canonical annotations.jsonl with real atomic labels/evidence/reviewer.
.venv/bin/python -m factuality score \
  --cases results/factuality/local/repro-biobart/cases.jsonl \
  --annotations results/factuality/local/repro-planned-review/annotations.jsonl \
  --out results/factuality/repro-human-metrics.json
```

To reproduce the selected agent examples, use `factuality.examples.selected_examples(cases, reviewed_at)` and validate with `score`; these exact-string examples intentionally fail if a changed model produces different text. Do not transfer them onto changed outputs or call them human review.

## Remaining work

A real human pilot, frozen human rubric, source-only expected-fact inventories, exhaustive claim annotations, adjudication and reviewer agreement remain undone. Anyone who has read the selected output examples should use an independent blind planner for those eight cases, or exclude them from unbiased omission rates. No empirical factuality error rates or model factuality ranking are justified yet. Validation selection bias, narrow automatic checks, source ambiguities, imperfect section mapping and model-visible truncation remain limitations. No mitigation was trained or claimed to solve hallucinations.

## Reproducing paired baseline exports

Both baselines use the same case JSONL and identical Flask input/decoding settings. The no-adapter run substitutes only the model-loading function; the actual Flask route, prompt construction and generation logic remain unchanged. BART-base and BioBART-large differ in size as well as pretraining, so these comparisons cannot isolate medical pretraining.

```bash
FACTUALITY_BART_PATH=$(.venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("facebook/bart-base", revision="aadd2ab0ae0c8268c7c9693540e9904811f36177", local_files_only=True))')
FACTUALITY_BART_ADAPTER_PATH=$(.venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("jonleed/clinical-note-bart-lora", revision="8a61695044124b479ba73da5134f064b5e36ae53", local_files_only=True))')
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m factuality generate \
  --cases results/factuality/local/repro-validation.jsonl \
  --out results/factuality/local/repro-bart-base --threads 2 \
  --base-model "$FACTUALITY_BART_PATH" --adapter none
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m factuality generate \
  --cases results/factuality/local/repro-validation.jsonl \
  --out results/factuality/local/repro-bart-lora --threads 2 \
  --base-model "$FACTUALITY_BART_PATH" --adapter "$FACTUALITY_BART_ADAPTER_PATH"
.venv/bin/python -m factuality summarize \
  --run results/factuality/local/repro-bart-base --out results/factuality/repro-bart-base
.venv/bin/python -m factuality summarize \
  --run results/factuality/local/repro-bart-lora --out results/factuality/repro-bart-lora
.venv/bin/python -m factuality paired \
  --runs results/factuality/local/repro-biobart \
         results/factuality/local/repro-bart-base \
         results/factuality/local/repro-bart-lora \
  --out results/factuality/repro-paired.json
```

`paired` refuses different cases, order, source hashes, prompts, generation settings, tokenization settings or actual input IDs. Per-case predictions remain local; public comparison artifacts contain hashes and separate overlap scores. Factuality comparisons remain pending until those same paired cases are actually annotated.
