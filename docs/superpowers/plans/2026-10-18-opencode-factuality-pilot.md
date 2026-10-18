# OpenCode Targeted Factuality Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task after user review. Use inline execution; do not spawn Codex implementers or reviewers. Steps use checkbox syntax for tracking.

**Goal:** Implement and demonstrate age/gender contradiction, unsupported-claim, and omission evaluation through a capped comparison of two free OpenCode models.

**Architecture:** Local fact inventories and deterministic demographic suggestions feed an evidence-grounded semantic reviewer. A separate OpenCode adapter enforces model and usage restrictions. A frozen 24-case synthetic pilot measures both reviewers without changing serving, training, historical results, or the reserved MTS test split.

**Tech Stack:** Existing Python `.venv`, standard-library persistence/process control/unittest, existing `factuality` validators, and installed `opencode2`.

**Spec:** [Approved design](../specs/2026-10-18-targeted-factuality-design.md).

## Global Constraints

- Only `opencode/deepseek-v4.1-flash-free` and `opencode/glm-5.3-flash-free` may receive inference requests. No Codex subprocesses, paid fallback, credential extraction, or third-party-client bypass.
- Maximum 48 inference attempts total, including failures and diagnostics; one request at a time; zero automatic retries.
- Enforce 2,000 output tokens per attempt, 250,000 total actual input tokens, 96,000 total actual output tokens, and a 120-second per-attempt deadline. If the installed client cannot enforce/audit the controls, stop before a batch.
- Disable tools, MCP, delegation, auxiliary title inference, warming, compaction, and fallback. Stop on quota/authentication errors or unexpected model calls.
- Freeze 24 cases before inference: six age cases, six gender/sex/pronoun cases, six unsupported-claim cases, six omission cases; three errors and three controls in each group.
- A full dialogue accompanies local facts. Never infer absence from extraction failure, silently truncate inputs, infer gender from stereotypes, or count failed/unassessed items as passes.
- Freeze source-supported, section-relevant expected facts before prediction assessment. Keep synthetic gold, model proposals, rule suggestions, and expert labels separate.
- Cache by exact evidence, prompt, model, configuration, and CLI identities; preserve reservations and attempts across restart. Stop after the pilot. Do not automatically review the 300 summaries or run another benchmark.
- Preserve the current checkout and uncommitted project work. Commit only this task's files; do not sweep unrelated changes into commits. No additional packages or model downloads are required.
- The reused `factuality` helpers are currently untracked. Before feature commits, preserve and add only the required baseline files (`__init__.py`, `io.py`, `demographics.py`, `checks.py`, `scoring.py`) in a clearly identified dependency commit; do not silently include historical transports, datasets or other uncommitted work.

## Review Focus

1. Hidden client requests or an unexpected default model must stop execution and consume ledger allowance rather than being omitted from usage reporting (Tasks 3–4).
2. Restart after dispatch but before a result is saved must retain the attempt/token reservation and must not repeat that case automatically (Task 3).
3. A relative's age, uncertain age, or repeated quotation must not become definitive patient evidence (Tasks 1 and 4).
4. Missing extraction and ambiguous judgments must remain visible in end-to-end denominators; a high score on a small assessed subset is not complete coverage (Tasks 4–5).
5. Optional/out-of-section facts and reference-only assertions must not become omission errors (Tasks 1–2 and 4).

## File and artifact boundaries

| File | Responsibility |
|---|---|
| `factuality/facts.py` | Normalized local fact records and source-only expected-fact freeze |
| `factuality/free_pilot_cases.py` | Deterministic 24-case construction and private gold manifests |
| `factuality/free_protocol.py` | Blinded payloads and exact-evidence response validation |
| `factuality/opencode_backend.py` | Read-only preflight, isolated client invocation, persistent budget and cancellation |
| `factuality/free_pilot.py` | Explicit prepare/preflight/run/report commands; no automatic execution on import |
| `factuality/free_report.py` | Coverage, four capability scores, disagreements and local review HTML |
| `factuality/prompts/free_inventory_v1.txt` | Frozen short evaluation rubric and output contract |
| `tests/test_free_*.py` | Offline regression controls; never call a provider |
| `results/factuality/local/opencode-pilot-2026-10-18/` | Private source/summary text, requests, events, ledger, responses and review HTML |
| `results/factuality/opencode-pilot-2026-10-18/` | Aggregate manifest, metrics, usage and verification only |
| `docs/opencode_factuality_pilot.md`, `README.md` | Execution record, results, limits and supported project description |

Do not modify `factuality/semantic_run.py` or resume its stopped study. Reuse pure helpers from `io.py`, `demographics.py`, `checks.py`, and `scoring.py`; no inference helpers from the historical transport are needed.

---

### Task 1: Local facts and section-aware expected facts

**Files:** Create `factuality/facts.py`; create `tests/test_free_facts.py`.

**Interfaces:**
- `extract_facts(text: str, *, dialogue: bool) -> list[dict]` returns subject, fact type, normalized value/unit, negation, certainty, temporal context, exact span, and extractor provenance. Unknown attributes are null/unresolved.
- `freeze_expected(dialogue: str, section: str, reference: str, candidates: list[dict]) -> dict` validates candidate source evidence, records required/optional/out-of-section status and provenance, and binds the entire source plan to a content hash. It never receives a prediction.
- `compare_demographics(case: dict) -> list[dict]` preserves the existing suggestions as proposals, without changing the current checker version or historical scores.

- [ ] Record the initial Git status and run `.venv/bin/python -m unittest discover -s tests -v`. Preserve those results. If the required baseline helpers remain untracked, commit exactly the five dependency files named in Global Constraints with message `chore: preserve reusable factuality evaluation core`; leave their content unchanged.
- [ ] **Write failing offline tests.** Use literal encounters to assert that patient age 57 is distinct from father age 67; age expressed as uncertain stays uncertain; a gestational week is a quantity; pronouns do not establish sex; an explicit medication dose/negation retains units and polarity. Freeze controls must reject missing source quotations, exclude maternal history from chief complaint, and preserve optional status even when a reference contains the fact.

```python
def test_age_is_attributed_to_patient_not_father(self):
    facts = extract_facts('Patient: I am 57 years old. My father is 67.', dialogue=True)
    self.assertIn(('patient', 'age', 57), [(f['subject'], f['fact_type'], f['value']) for f in facts])
    self.assertNotIn(('patient', 'age', 67), [(f['subject'], f['fact_type'], f['value']) for f in facts])
```

- [ ] Run `.venv/bin/python -m unittest discover -s tests -p 'test_free_facts.py' -v`; verify the new tests fail because the implementation is absent.
- [ ] Implement the interfaces using existing demographic extraction and finite explicit quantity/medication/negation patterns. Keep unsupported constructions unresolved; do not add an external NER model or pretend the local inventory is exhaustive.
- [ ] Rerun the same command; require all new tests to pass, then run existing demographic tests to establish unchanged behavior.
- [ ] Commit only the Task 1 source/test files with message `feat: add source-grounded local fact inventories`.

### Task 2: Frozen pilot cases and blinded response protocol

**Files:** Create `factuality/free_pilot_cases.py`, `factuality/free_protocol.py`, `factuality/prompts/free_inventory_v1.txt`, `tests/test_free_protocol.py`.

**Interfaces:**
- `build_cases() -> list[dict]` returns the 24 controlled cases with section, source, reference, prediction, source-only expected facts and private target gold. Gold contains capability, error/control status, target type/span or expected-fact ID, and the correct semantic label.
- `freeze_pilot(out: Path) -> dict` writes new-only JSONL/manifests and hashes cases, gold, expected plans and prompt. It refuses to overwrite a different freeze.
- `make_payload(case: dict, source_plan: dict) -> dict` exposes dialogue, requested section, prediction, local fact proposals and source-plan facts. It excludes gold, target spans, fixture family, source-model identity and other reviewers' results.
- `validate_response(payload: dict, raw: dict) -> dict` validates exhaustive claim attestation, atomic claim labels/quotes and every expected-fact ID; supported/contradicted claims need source evidence, present facts need generated evidence, optional facts are excluded, and ambiguity stays visible.

- [ ] **Write failing tests** for a fabricated quotation, cross-patient evidence, repeated evidence, duplicate/missing expected-fact IDs, absent claim extraction in a nonempty note, and a harmless paraphrase marked present. Verify payload blinding by passing a case with private sentinel fields and checking the emitted payload contains none of them. Verify source-plan hashes change when evidence/requiredness changes.
- [ ] Run `.venv/bin/python -m unittest discover -s tests -p 'test_free_protocol.py' -v`; confirm missing-module/function failures.
- [ ] Implement the fixed case builder and protocol. Reuse suitable existing challenge constructions; add controlled omission/control pairs. Freeze expected facts before constructing assessment payloads. The model extracts all claims, including facts the local table missed; it does not receive the private target.
- [ ] Run the same tests to green. Manually inspect all 24 construction targets against their exact source/prediction strings, record the synthetic-author review, and freeze them before inference. Never label this step physician review.
- [ ] Commit the Task 2 files with message `feat: freeze blinded targeted factuality pilot`.

### Task 3: Free-only OpenCode transport and durable limits

**Files:** Create `factuality/opencode_backend.py`, `tests/test_free_backend.py`.

**Interfaces:**
- `preflight(executable: str = 'opencode2') -> dict` performs metadata/configuration checks only. Return CLI version, catalog availability, effective policy/configuration hashes and `ready`/`blocked` with reasons. No model request is allowed here.
- `BudgetLedger(path: Path)` provides `reserve(job_id: str, model: str, input_bound: int) -> str` and `settle(reservation_id: str, usage: dict, status: str) -> None`. Reservations include one attempt and 2,000 output tokens; unsettled reservations remain charged. Refuse duplicate jobs, paid IDs, concurrent reservations or insufficient allowance. Serialize updates with a local lock and atomic writes.
- `run_review(job: dict, model: str, ledger: BudgetLedger, out: Path) -> dict` runs one fresh context with explicit isolated configuration, records raw events and actual model/usage metadata, validates the response, and returns complete/failed/blocked. No retries or model substitution.

- [ ] **Write failing tests** using a local fake CLI process with event fixtures, not a network mock. Demonstrate that a paid/default-model event is rejected, an auxiliary response is counted and stops the run, a 49th reservation is refused, insufficient token budget prevents process launch, cancellation kills a child process, and an interrupted reservation remains charged after reopening the ledger.

```python
def test_restart_does_not_repeat_an_unsettled_job(self):
    ledger = BudgetLedger(self.ledger_path)
    ledger.reserve('case-01:deepseek', 'opencode/deepseek-v4.1-flash-free', 1000)
    reopened = BudgetLedger(self.ledger_path)
    with self.assertRaises(ValueError):
        reopened.reserve('case-01:deepseek', 'opencode/deepseek-v4.1-flash-free', 1000)
```

- [ ] Run `.venv/bin/python -m unittest discover -s tests -p 'test_free_backend.py' -v`; observe the expected missing implementation failures.
- [ ] Implement preflight against the installed v2 schema/help and read-only configuration output. Never print credential values. Verify actual effective restrictions, not just text in a proposed config. If the preview cannot enforce output/step limits, disable auxiliary inference, bound overhead, or expose auditable metadata, return blocked before running a batch.
- [ ] Implement owned process-group lifecycle, immutable request/event records, pre-dispatch reservation and post-response settlement. Hash CLI, model alias, prompt, task and effective configuration for cache identity. A requested alias is not an exposed provider weight revision; keep unknown revisions null.
- [ ] Run the backend tests to green; inspect the fake CLI cancellation result and durable ledger totals. No real inference during this task.
- [ ] Commit with message `feat: add capped free-only OpenCode review transport`.

### Task 4: Pilot orchestration and dry-run entry point

**Files:** Create `factuality/free_pilot.py`, `tests/test_free_pilot.py`.

**Interfaces:** Provide `python -m factuality.free_pilot prepare|preflight|run|report --out <directory>`. `prepare` freezes data without inference; `preflight` uses Task 3; `run` requires the existing frozen manifest and ready preflight; `report` is offline. Default invocation/help never starts inference.

- [ ] **Write failing tests** for an interrupted run, a changed prompt/freeze/config on resume, unsupported CLI preflight, quota failure stopping subsequent dispatch, and normal completion stopping after 48 attempts. Run the real orchestrator against the fake CLI and check filesystem results and attempted-job counts.
- [ ] Run `.venv/bin/python -m unittest discover -s tests -p 'test_free_pilot.py' -v`; observe expected missing implementation failures.
- [ ] Implement deterministic case order with alternating first model by case index, one request at a time. Inspect actual metadata from the first authorized case before dispatching the second. Catch transport/protocol failures as failed assessments; stop immediately for policy, usage, quota, authentication or cancellation errors. Resume only untouched jobs using the same durable ledger; do not retry failed/uncertain jobs automatically.
- [ ] Require the full prompt plus dialogue to fit the conservative reservation. Oversized cases become unassessed; no truncation. Validate and preserve every response before scoring. No callback or import may launch the historical semantic study or production evaluation.
- [ ] Run the same tests to green and inspect `--help` plus an offline prepare/preflight rehearsal. A blocked preflight is a valid reported outcome, not a reason to switch models/executables.
- [ ] Commit with message `feat: orchestrate bounded targeted factuality pilot`.

### Task 5: Coverage-aware comparison and review report

**Files:** Create `factuality/free_report.py`, `tests/test_free_report.py`; update `docs/opencode_factuality_pilot.md` and `README.md`.

**Interfaces:** `build_report(frozen: dict, results: list[dict], ledger: dict) -> dict` returns per-model, per-capability confusion counts, precision/recall/F1, false-positive counts, assessed/valid/extracted coverage, end-to-end target correctness, disagreements, token totals and elapsed time. Missing positive targets are false negatives. `write_report(report: dict, out: Path) -> None` writes aggregate JSON and a local escaped HTML evidence report.

- [ ] **Write failing tests with literal expected counts.** Two positive and two negative targets with predictions error/missing/error/correct must yield TP=1, FN=1, FP=1, TN=1, precision=0.5 and recall=0.5. A correct label on the wrong fact type is an extraction miss. An omitted optional fact must not increase the omission-error count. Failed controls remain unassessed rather than successful true negatives. HTML must escape embedded markup from dataset/model text.
- [ ] Run `.venv/bin/python -m unittest discover -s tests -p 'test_free_report.py' -v`; confirm missing implementation failures.
- [ ] Implement target matching against exact generated spans/type or frozen expected-fact IDs. Keep assessed-only confusion counts alongside full denominators and missing counts. Report missing actual usage as unknown and block a usage-compliance claim.
- [ ] Implement the design's provisional ranking: mean four capability F1s; ties use fewer false positives, higher valid coverage, fewer actual tokens, then stable model-ID order. Always expose six-case-per-capability limits; do not recommend production solely from the rank or describe model consensus as gold.
- [ ] Run report tests and `.venv/bin/python -m unittest discover -s tests -v`; require all offline tests to pass. Add documentation with runnable commands and pending execution status, preserving historical reports and the user's exact intended project wording as a completion target.
- [ ] Commit with message `feat: report targeted factuality performance and coverage`.

### Task 6: Execute only the approved pilot and publish its actual outcome

**Files:** Write the private/public artifact locations listed above; finalize `docs/opencode_factuality_pilot.md` and the README execution status.

- [ ] Run the full offline suite once more after final edits; save command, environment and test counts to the new verification artifact.
- [ ] Run `.venv/bin/python -m factuality.free_pilot prepare --out results/factuality/local/opencode-pilot-2026-10-18` and inspect the frozen manifest and hash bindings without exposing gold to a provider.
- [ ] Recheck official free-model availability and run `.venv/bin/python -m factuality.free_pilot preflight --out results/factuality/local/opencode-pilot-2026-10-18`. If blocked, write the exact reason, zero inference count, and required remedy; stop here.
- [ ] If ready, run `.venv/bin/python -m factuality.free_pilot run --out results/factuality/local/opencode-pilot-2026-10-18`. The first real assessment is part of the 48-attempt budget. The adapter must stop if observed metadata invalidates the preflight assumptions. Respect all early-stop conditions; do not fill missing responses with other models.
- [ ] Run the offline `report` command. Verify both successful and stopped outcomes, reconcile ledger attempts with raw model events, and confirm no owned OpenCode/Codex review subprocess remains active.
- [ ] Write actual results, examples and limits. State which of the four evaluator paths were demonstrated and whether either model merits further validation. Do not promote the proposed project description to a completed claim if preflight blocked or a path was not demonstrated. Do not begin the 300-summary review.
- [ ] Commit only implementation, aggregate results, and documentation belonging to this plan. Keep raw dataset text and review queues in the ignored local directory. Report measured usage, coverage, tests, failures and the next bounded decision to the user.

## Plan self-review and handoff

- [x] Every design requirement maps to a task; the reserved test split, prior artifacts and usage restrictions are preserved.
- [x] All interfaces used downstream are defined above; no code or inference is required to review this plan.
- [x] The five Review Focus failures each have an explicit owning test.
- [x] The live step can end blocked or partial and must still produce an honest report; there is no automatic production stage.
- [ ] User reviews this implementation plan and confirms execution method.

Recommended method: inline execution with `superpowers:executing-plans`, without Codex subagents or additional model code-review batches. The user’s usage constraint takes precedence over a skill's default recommendation to spawn reviewers. The only external inference is the bounded OpenCode pilot.
