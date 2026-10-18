# Targeted factuality pilot — October 18, 2026

The targeted evaluation harness is implemented, but **the semantic run is blocked by OpenCode's free-tier access check**. Do not describe this run as a completed LLM evaluation or clinical validation.

The selected reviewer is `opencode/deepseek-v4.1-flash-free`. The user requested a single researched choice rather than a model comparison. DeepSeek's [published model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) supports it as a reasonable general reasoning model; it does not demonstrate superiority on this particular clinical factuality task. GLM was not evaluated.

## Actual results

| Check | Result |
|---|---|
| Frozen controlled cases | 24: six each for age, gender/sex/pronouns, unsupported claims, and omissions |
| Errors and controls | Three errors and three controls per capability |
| Local demographic rule diagnostic | 6/6 known contradictions flagged; 0/6 controls flagged as contradictions |
| OpenCode execution attempts | 5, all failed before a usable review |
| Valid semantic reviews | **0/24** |
| Remaining cases | 19 not attempted; four earlier failures and one separately authorized terminal probe |
| Semantic unsupported-claim / omission accuracy | **Not measured** |
| Actual provider token counts | Not available for the failed attempts; full reservations retained |
| Reserved token allowance for these attempts | 50,000 input / 10,000 output upper bounds; not a claim of actual consumption |
| Paid fallback / Codex review subprocesses | None in this pilot |
| Offline regression suite | 109 tests passed; see the versioned verification artifact |

The local rule result is a diagnostic on small, implementer-authored fixtures. It does not estimate clinical prevalence or independently validate general demographic accuracy. Missing semantic assessments remain in coverage denominators. Any false-negative counts in the machine report describe unassessed pilot targets, not observed failures of DeepSeek's reasoning.

## What is implemented

- Finite local fact extraction and separate rule proposals, with patient/relative attribution and explicit evidence spans.
- Blinded semantic payloads and exact quotation validation; unsupported and contradicted labels remain distinct.
- Frozen source-only expected facts, with section eligibility, optional exclusions, and preserved original model proposals.
- A free-model-only OpenCode adapter, durable reservations, no automatic review retries, cancellation, and a persistent provider-denial barrier.
- Offline prepare/report commands and local HTML inspection of all 24 cases. Importing the modules does not start inference.

## Access failure and diagnosis

1. Preview `opencode2 v0.0.0-beta-18743` initially failed to register the restricted agent in a short-lived private server.
2. A native one-shot API route returned the free-tier restriction and was abandoned.
3. Waiting for private-server initialization made the normal restricted client runnable. Both that preview and installed stable `opencode v2.0.32` then returned HTTP 403:

   > OpenCode's free tier can only be used from within OpenCode.

The stable client, private configuration, selected free model, zero catalog pricing, denied tools, output cap, and single-step reviewer were verified through the client's metadata endpoints. The model being in the catalog did not establish that a request would be accepted.

Matching upstream reports exist for [official 2.0.32 clients](https://github.com/anomalyco/opencode/issues/56616) and [restricted custom agents](https://github.com/anomalyco/opencode/issues/50627). Those are reported issues, not proof of the precise cause on this machine. No authentication headers were forged, no credentials were extracted, and no paid model was substituted. Tools were not re-enabled to evade the check. The user's existing background service was not restarted.

The run is persistently blocked until a working authorized provider/client route is established. Re-running `run` against this ledger returns the block without another inference request. No automation, polling for a reset, or background retry was installed.

## Reproducibility and inspection

```bash
# Offline suite; never calls a provider.
.venv/bin/python -m unittest discover -s tests -v

# Offline report from saved attempts; never calls a provider.
.venv/bin/python -m factuality.free_pilot report \
  --out results/factuality/local/opencode-pilot-2026-10-18
```

Public aggregate artifacts: [execution status](../results/factuality/opencode-pilot-2026-10-18/execution.json), [semantic coverage](../results/factuality/opencode-pilot-2026-10-18/metrics.json), [local rule diagnostic](../results/factuality/opencode-pilot-2026-10-18/local_rule_metrics.json), and [verification](../results/factuality/opencode-pilot-2026-10-18/verification.json).

Private text-bearing artifacts remain ignored under `results/factuality/local/opencode-pilot-2026-10-18/`: frozen cases and expected facts, blinded prompt files in `review-packet/`, requests, raw events, owned-session exports, ledger, and `review.html`. Existing predictions, training/serving, historical results, and the reserved MTS test split were not regenerated or overwritten.

## Supported project description

The implementation targets the user's description:

> Built a targeted factuality evaluation for age and gender contradictions, unsupported claims, and omissions in clinical summaries.

For this execution, qualify that the harness and local diagnostics are complete while **LLM execution and semantic performance validation remain blocked**. Expert-equivalent accuracy, clinical safety, complete omission recall, and review of all 300 saved summaries are not established.

## Follow-up: authorized full terminal probe

After the user confirmed that ordinary OpenCode terminal sessions work, one frozen case was submitted through the official full-screen terminal interface in a pseudo-terminal. The selected DeepSeek model, custom reviewer, and deny-all tool policy were unchanged. The interface launched and displayed the supplied case, then displayed the same free-tier access rejection. The terminal was closed at its 60-second deadline and its owned private server was stopped. No review was produced.

The capture remains private in `terminal-probe/terminal.raw`. Temporary-path spelling (`/var` versus `/private/var`) prevented automatic session selection for export; the terminal capture independently records the provider rejection. This was a throwaway feasibility probe, not a completed reusable terminal adapter. The original provider barrier remains active, and all five attempts retain their full token reservations.

OpenRouter is the proposed next transport, pending a locally configured API key. Its [live catalog](https://openrouter.ai/api/v1/models) lists `nvidia/nemotron-3-ultra-550b-a55b:free` at zero prompt and completion prices. NVIDIA reports instruction-following and non-hallucination results in its [technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf), making this a reasonable candidate for evidence-grounded review; that is an engineering inference, not proof of clinical factuality accuracy or superiority on these cases. DeepSeek V4.1 Flash was not listed as a free OpenRouter variant at the time of inspection. No OpenRouter inference has been attempted and no paid route has been selected. Preserve the four-endpoint rubric and frozen cases; report any provider/model change separately.
