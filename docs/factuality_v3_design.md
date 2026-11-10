# Factuality evaluation v3 — design

Date: 2026-11-09. Status: design as implemented. Changes made during the run (packet size, held-out top-up, scoring fixes) are described in the [v3 report](factuality_v3_report.md).
Supersedes nothing: v2 artifacts, reports and the 2026-10-18 pilot stay frozen as historical records.

## 1. Intent

**Purpose.** Make the README claim — a targeted factuality evaluation for **age, gender and race/ethnicity contradictions, unsupported claims and omissions** in clinical summaries — honest and defensible to a technical reviewer, with the rigor expected of a project touching medicine, even though it is a personal project.

**Success criteria.**

- Every headline number comes from measurement against independent human labels (not from constructed or implementer-authored gold), with a dialogue-cluster 95% CI and its coverage/abstention rate.
- No detector ever receives another detector's output, a plan, construction labels or model identity.
- Every issue in the 2026-11-09 audit (§2) is resolved by design, not by rewording.
- Race/ethnicity insertion is checked on **every** summary, regardless of whether the source mentions race.

**Constraints and decisions.**

| Decision | Choice |
|---|---|
| Ground truth | The project owner as a single, blinded, non-clinician annotator; disclosed as such |
| Primary judge | Claude Sonnet 5.5 via the owner's subscription through headless Claude Code (`claude -p`) |
| Secondary judge | Existing pinned free Nemotron 3 Ultra (OpenRouter) for agreement only |
| Data | 100 MTS-Dialog validation dialogues plus programmatic counterfactual edits of them. MTS-Dialog test splits stay untouched. |
| Architecture | Approach A: two independent detectors (rules R, LLM judge J), both scored against human labels |

## 2. Audit issues and their resolution

| # | Audit finding (2026-11-09) | Resolution in v3 |
|---|---|---|
| 1 | Pilot payload included `rule_proposals` with labels; model copied them (8/8 when cued, 0/4 when not) | Judge payload is an explicit field allowlist; no rule output, plans, provenance or IDs. Enforced by test (§7). |
| 2 | BART-base (no adapter) outputs are prompt + dialogue echoes; screening compared source to itself | Output-validity gate (§3.2) classifies non-summaries before any scoring; reported as a validity rate only. |
| 3 | Sex claims 100% unclear, pronouns ~70% unclear on real data | v3 summary patterns parse descriptor-interleaved sex claims; new `cross_dimension_inconsistency` issue; abstention reported as coverage (§4). |
| 4 | Pilot: 19 one-line implementer-authored cases, prompt revised mid-run, no CI | Calibration/held-out split of human-labeled real summaries plus ~250 counterfactual cases; prompt frozen before held-out; cluster bootstrap CIs (§5, §6, §7). |
| 5 | Omission plans handed the reviewer the exact fact quote | Judge builds its own section inventory without seeing plans; matched to human plans afterwards (§6.4). |
| 6 | Case-06 relative-age fact-type convention counted as a factuality failure | Relative ages are `fact_type: age, subject: relative`; type-convention mismatches scored separately from label errors (§4.3). |
| 7 | Unsupported claims and omissions never assessed on real summaries | Judge runs on all 200 real fine-tuned summaries; human labels on 100 of them (§5). |
| 8 | `history of present illness` vs `History of Present Illness` split sections | Case export normalizes section names to canonical lowercase; original header kept in provenance (§3.1). |

## 3. Architecture and data flow

All artifacts for v3 live under `results/factuality/local/v3/` (git-ignored) with hash-bound manifests; public aggregates go to `results/factuality/v3-<date>/`. Each stage consumes the frozen output of the previous stage and never modifies it.

```
cases ─► validity gate ─► ┬─► human source-only plans (frozen) ─► human claim labels ─┐
                          ├─► Detector R (rules v3) ───────────────────────────────────┼─► scoring ─► reports
                          └─► Detector J (LLM judge) ───────────────────────────────────┘
counterfactual packet ─► Flask generation ─► validity gate ─► R, J ─► construction-gold scoring
```

### 3.1 Case export (existing `export.py`, modified)

- Normalize `requested_section` to canonical lowercase; keep the original header in `source.original_headers`.
- Cohort: the existing 300 saved cases (100 dialogues × BioBART-large+LoRA, BART-base+LoRA, BART-base) plus counterfactual cases (§5.1).

### 3.2 Output-validity gate (new `factuality/validity.py`)

Classifies each prediction as exactly one of:

- `prompt_echo` — prediction contains the prompt prefix `Summarize the following doctor-patient dialogue`, **or** ≥ 0.6 of the prediction's word 8-grams occur verbatim in the dialogue.
- `empty` — no non-whitespace characters after stripping the section label prefix.
- `degenerate` — any single word 4-gram accounts for ≥ 30% of the prediction's 4-grams.
- `summary` — none of the above.

Thresholds are fixed here and tested against fixtures shaped like the real BART-base outputs and the real fine-tuned outputs (all 200 fine-tuned outputs must classify as `summary`; all 100 BART-base outputs must classify as `prompt_echo`; any exception is investigated and the threshold change recorded before scoring). Only `summary` outputs enter factuality denominators. Validity rates are reported per model.

## 4. Detector R: rule checker `patient_demographics_v3`

New version; `patient_demographics_v2` code path and saved results remain unchanged. All v2 tests continue to pass against v3 unless a test is explicitly updated with a recorded behavior change.

### 4.1 Race/ethnicity dimension (new `fact_type: race_ethnicity`)

**Vocabulary** (fixed table, normalized values; synonyms map to one value):

| Normalized value | Accepted surface terms |
|---|---|
| `white` | white, Caucasian |
| `black` | Black, African American, African-American |
| `hispanic_latino` | Hispanic, Latino, Latina, Latinx |
| `asian` | Asian, Asian American, and explicit subgroup adjectives listed in code (e.g. Chinese American, Korean American, Vietnamese American, Filipino, Indian American) |
| `american_indian_alaska_native` | American Indian, Native American, Alaska Native |
| `nhpi` | Native Hawaiian, Pacific Islander |
| `mena` | Middle Eastern, North African, Arab American |
| `multiracial` | multiracial, biracial, mixed race |

**Summary-side claims** are extracted only from patient-anchored constructions: `(this is|the patient is) (a|an) [N-year-old] <race> (male|female|man|woman|…)`, `the patient is <race>`, `<N-year-old> <race> (male|female|…) (who|with|presents)`. Excluded explicitly: `white blood cell(s)`, `white count`, `WBC`, `black stool(s)`, `blackout`, `black eye`, any race term in a clause whose subject is a relative/provider (reuse `OTHER_PERSON`).

**Source-side evidence**: patient self-statements (`I am / I'm <race>`, `I identify as <race>`), clinician statements about the patient (`the patient is <race>`, `this is a … <race> …`), and an answer to a preceding race/ethnicity question (`what is your race`, `ethnicity`, `how do you identify`). Names, language spoken, nationality/country of origin and location are **never** race evidence; their presence is recorded as `proxy_cue` for review.

**Proposed labels** for each summary race claim:

| Full-source evidence | `proposed_label` | `issue` |
|---|---|---|
| Patient-attributed race matches | `supported` | `demographic_comparison` |
| Patient-attributed race differs | `contradicted` | `demographic_comparison` |
| No vocabulary term anywhere in the full source and no proxy cue | `unsupported` | `race_inserted` |
| Vocabulary term present but not patient-attributed, conflicting, qualified, or only proxy cues | `unclear` | `race_not_established` |

Model-visible labels are computed the same way over `model_input.text`. A full-source `supported` race outside the visible input is `visibility: truncated`, as for other dimensions.

`unsupported` is a **new automatic proposal label** (v2 emitted only supported/contradicted/unclear). It is used for race only, because the absence test is a full-source vocabulary scan rather than an extractor failure.

### 4.2 Sex/pronoun coverage recovery

- Summary-side sex pattern accepts any sequence of race vocabulary terms and up to two other adjectives between the age construction and `male/female/man/woman`.
- New issue `cross_dimension_inconsistency`: when a summary sex claim's implied pronoun class conflicts with **patient-attributed** source pronoun evidence (from `pronoun_evidence`), emit `proposed_label: unclear`, `review_required: true`. These are reported as a separate "gender-reference inconsistency" category, never as sex contradictions, because pronouns are not sex evidence.
- Honorific-only conflicts remain `unclear` with issue `gendered_address_inconsistency` (unchanged).

### 4.3 Rubric change: relative ages

Relative/third-party ages are `fact_type: age` with `subject: relative`. Scoring of any detector distinguishes `label_correct` from `type_convention_match`; a correct label with a mismatched type convention counts as label-correct and is reported in a separate convention-mismatch count.

## 5. Data: counterfactual packet and human annotation sample

### 5.1 Counterfactual packet (new `factuality/counterfactual.py`)

Each edit applies to one validation dialogue, is stored as an exact, replayable diff, and is placed so the edited text falls within the first 512 model tokens (verified with the real tokenizer). Each edited dialogue is rejected if the edit creates a conflict with another patient-attributed statement of the same dimension (checked by R's extractors).

| Family | Edit | Construction gold for the edited attribute |
|---|---|---|
| Race injection | Insert `Patient: I'm <surface term>.` or a clinician intro with race, after the first doctor turn; values rotated evenly across the 8 groups using a seeded order | Summary race claim must match or be absent; a different value is `contradicted`; absent is `omitted_optional` |
| Race removal | Delete the race term from the 3 validation dialogues stating race (val:14, val:43, val:81), adjusting grammar by template | Any summary race claim is `unsupported` |
| Age shift | Replace the patient's explicit age with age + 7 (dialogues where R extracts exactly one patient age) | Summary age must equal the edited value |
| Pronoun/honorific swap | Consistently swap patient-attributed pronouns and honorifics (dialogues where R extracts patient pronoun evidence) | Summary pronouns/sex terms must follow the swap or be absent |

Target size ≈ 250 edited dialogues (100 race injection, 3 race removal, remaining split between age shift and pronoun swap by eligibility). Each is generated with BioBART-large+LoRA and BART-base+LoRA through the existing Flask `generate` path and settings, run in the background with a manifest (≈ 500 CPU generations). Construction gold covers only the edited attribute; other claims in these summaries are not scored against construction gold. Per-race results are reported as underpowered and are not a fairness claim.

### 5.2 Human annotation sample

- **Population:** 200 real fine-tuned summaries (BioBART-large+LoRA, BART-base+LoRA) that pass the validity gate.
- **Size:** 50 dialogues × both models = 100 summaries. Estimated workload: 50 source-only plans + 100 claim reviews.
- **Selection:** 40 dialogues by `random.Random(20261109).sample` over the 100 validation dialogue IDs (sorted), then 10 enrichment dialogues chosen from the remaining 60 whose v3 output contains a `contradicted`, `race_ethnicity`, or `cross_dimension_inconsistency` suggestion (seeded random choice if more than 10 qualify; all if fewer). Inclusion probabilities are recorded. Enrichment dialogues contribute to detector sensitivity only, never to prevalence estimates.
- **Held-out top-up (owner decision 2026-11-09):** only 4 dialogues qualified for enrichment and all 4 were already seen (so calibration-only). Held-out is therefore topped up to 35 with a separate seeded draw (`seed + 1`) from unsampled, unseen dialogues; the original random/enrichment/calibration draw is unchanged. Top-up dialogues are random draws and count toward prevalence. Final sample: 40 random + 4 enrichment + 6 top-up = 50 dialogues (100 summaries).
- **Split (by dialogue, both models' summaries together):** 15 calibration dialogues, 35 held-out dialogues. Dialogues the owner has already seen outputs for (val:9, 14, 24, 37, 39, 43, 59, 81) may be placed in calibration only and are barred from held-out.
- **Blinding:** source-only plans for all 50 dialogues are frozen (hash recorded) before any prediction is opened. Predictions are presented in seeded shuffled order with model identity hidden. R and J outputs are hidden from the annotator until that case's labels are finalized.
- **Tooling:** extend the existing read-only HTML review (`review.py`) into a local annotation page writing canonical annotation JSONL (schema v1 plus `race_ethnicity` and `subject`). The annotator records `reviewer_kind: human` and attests blindness per the existing protocol.

## 6. Detector J: LLM judge

### 6.1 Judges

- **Primary:** `claude-sonnet-5-5` via headless Claude Code (`claude -p`) under the owner's subscription. Runs on all 200 real fine-tuned summaries, all counterfactual summaries and the annotation sample.
- **Secondary:** existing pinned free `nvidia/nemotron-3-ultra-550b-a55b:free` (zero-price guards retained, new separately declared ledger), run on the 35 held-out dialogues only, for agreement reporting.

### 6.2 Payload allowlist

The judge sees exactly: opaque case ID, full dialogue, model-visible input text, `truncated` flag, requested section, prediction, and the rubric. Any other key fails a test. No rule output, plans, provenance, construction labels or model identity.

### 6.3 Output contract

JSON validated by `--json-schema` and again locally:

1. `claims`: every atomic claim with `text`, `fact_type` (adds `race_ethnicity`), `subject` (`patient|relative|other`), `label` (`supported|contradicted|unsupported|unclear`), exact `generated_quote`, exact `evidence_quotes`, `rationale`, and `visibility` (`visible|truncated`).
2. `section_inventory`: source facts the judge considers in-section, each with exact source quote and `present|omitted` plus a summary quote when present.

Quotes are validated as exact substrings. Invalid output is recorded as a failed assessment, not dropped.

### 6.4 Matching judge output to human labels

- Claims: match on overlapping `generated_span` and same `subject`. Unmatched human claims → judge misses. Unmatched judge claims → reported as "extra atomization", not errors.
- Omissions: judge inventory facts matched to human plan facts by overlapping source span; judge omission recall measured against human `omitted` facts.

### 6.5 Claude Code backend (new `factuality/claude_code_backend.py`)

Invocation per case: one isolated process,
`claude -p --model claude-sonnet-5-5 --output-format json --json-schema <schema> --tools "" --strict-mcp-config --disable-slash-commands --no-session-persistence --setting-sources project --system-prompt <rubric>`
run with an empty scratch working directory (so the `project` source contains no settings, and user-level settings, hooks and plugins are excluded). The isolation probe below verifies this rather than assuming it. `--bare` is **not** usable because it forces API-key auth.

**Isolation probe (gate before any judge run, repeated whenever `claude --version` changes):** run a canary invocation and record the context the judge received (system-prompt snapshot and a canary self-report). Fail if any hook output (e.g. SessionStart plugin text), skill/plugin listing, CLAUDE.md, memory or MCP tool appears. Record CLI version, model, flags and probe hash in the run manifest.

**Determinism:** temperature cannot be set. Each held-out case gets **2 independent trials**; trial disagreement on a claim makes that claim `unclear` (never "pick the better one"); intra-judge agreement is reported. Calibration and counterfactual cases get 1 trial.

**Retries:** at most one automatic retry for transport failures (non-zero exit without JSON, timeout); every attempt is logged. Content/validation failures are never retried.

**Ledger:** declared cap of 1,000 Claude Code calls for the whole v3 run, reserved before each call. Usage-limit or rate-limit responses stop the run and record the stop reason; resuming skips cases with a recorded attempt and never re-sends completed cases.

**Terms check:** before the first judge run, confirm that personal batch evaluation through headless Claude Code is within the subscription's current usage terms. If not, stop; the fallback (Anthropic API with a $20 hard cost ledger) requires the owner's explicit approval.

### 6.6 Freezing

Prompt/rubric changes are allowed only after inspecting calibration-split cases and synthetic fixtures. Before the held-out run, freeze prompt hash, schema hash, model ID, CLI version and flags in a manifest. Any later change invalidates the held-out set for that judge configuration.

## 7. Metrics, reporting and testing

### 7.1 Metrics

Per summarizer, per detector, per split; full-source and model-visible views where applicable.

- **Validity rate** per model (§3.2).
- **Detector quality vs human labels (held-out):** precision, recall, F1 per error type — age, sex, pronoun, race/ethnicity, cross-dimension inconsistency, unsupported claim, omission — with 95% CIs from a dialogue-cluster bootstrap (2,000 resamples, seed 20261109). Abstention/`unclear` rates reported alongside; abstentions stay in denominators as non-detections.
- **Summarizer error rates:** contradiction, unsupported, omission rates per model from the 40 random dialogues only, with cluster bootstrap CIs.
- **Counterfactual:** per edit family, faithful-copy / contradicted / inserted / omitted rates per model; race injection broken down by group (labelled underpowered).
- **Agreement:** Claude trial-vs-trial, Claude vs Nemotron, each judge vs human; Cohen's kappa when each cell count ≥ 5, otherwise raw agreement only.

### 7.2 Predeclared reporting rules (fixed before held-out)

All results are reported regardless of outcome. Wording in the README is constrained:

- A detector may be described as "detects <error type>" only if, on held-out, it has ≥ 10 human-labeled positives and both precision and recall point estimates ≥ 0.80. Otherwise it is described as "screens for <error type>, measured recall X (95% CI a–b)".
- An error type with < 5 human-labeled positives on held-out is reported as "insufficient positives to estimate detector recall".
- Every headline includes "single non-clinician annotator, 35 held-out dialogues".

### 7.3 Reporting

- New public aggregates in `results/factuality/v3-<date>/` (no raw text). v2 snapshots, the 2026-10-18 pilot aggregates and their docs are left unchanged.
- New report `docs/factuality_v3_report.md`; README factuality section rewritten from v3 measurements only.
- Raw dialogues, predictions, human labels and judge responses remain under git-ignored `results/factuality/local/v3/`.

### 7.4 Testing (offline, test-first)

- R v3: race inclusion/exclusion fixtures (WBC, black stools, relatives, nationality-only, names), cross-dimension cases modelled on val:43/val:81 patterns (synthetic text, not copied dialogue), relative-age subject; all v2 regression tests.
- Validity gate: echo/empty/degenerate/summary fixtures.
- Counterfactual: exact diff reconstruction, edit inside first 512 tokens, conflict rejection, seeded rotation.
- Sampling/splitting: deterministic selection, seen-dialogue exclusion from held-out, no dialogue in two splits.
- Judge payload: allowlist test; blinding test that fails if any rule/plan/provenance field is present.
- Claude Code backend: fake `claude` executable for success, invalid JSON, timeout, usage-limit stop, retry-once, resume-without-resend, isolation-probe failure.
- Scoring: matching rules (§6.4), cluster bootstrap determinism, reporting-rule wording.
- No test performs real inference or network access.

## 8. Phasing

1. **Measurement integrity:** section normalization, validity gate, rubric relative-age change.
2. **Checker v3:** race/ethnicity, sex/pronoun recovery, rerun R on the 200 valid real summaries.
3. **Counterfactual packet:** edits, generation (background), R on outputs.
4. **Human annotation:** sampling/splitting, plan freeze, annotation tooling, owner labels (calibration first).
5. **Judge:** Claude Code backend + isolation probe + terms check, calibration runs, freeze, held-out runs (Claude ×2, Nemotron ×1), all real and counterfactual runs.
6. **Scoring and reporting:** metrics, v3 report, README rewrite.

Phases 2–3 can proceed while the owner annotates in phase 4, provided the annotator does not view R output for sampled dialogues.

## 9. Out of scope

- Clinician annotation, multiple-annotator agreement, or clinical-accuracy claims.
- Using MTS-Dialog test splits.
- NLI-model detector (approach C).
- Changing summarizer training or serving behavior.
- Fairness claims about differential summarizer behavior by race.
