"""Judge runner (spec §6.4–6.6): opaque IDs, per-trial jobs, resume, freeze gate, trial merge.

Per-trial records never contain real case IDs or run names; only key.json maps
opaque IDs back. Held-out runs refuse to start unless the frozen rubric/schema
hashes match the current files.
"""
import json
import time
from pathlib import Path

from .claude_code_backend import MODEL, CallLedger, build_command
from .io import digest
from .judge import (OUTPUT_SCHEMA, JudgeOutputError, build_payload, rubric_hash, rubric_text, schema_hash,
                       validate_judge_output)

STOP_STATUSES = ('usage_limited', 'auth_failed')
MAX_CONSECUTIVE_FAILURES = 5  # a broken prompt/CLI must not burn the call cap


def opaque_id(case_id, run):
    return digest(f'{case_id}|{run}')[:12]


def _load_json(path, default):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


def _check_freeze(freeze_manifest, cli_version_now=None):
    if freeze_manifest is None or not Path(freeze_manifest).exists():
        raise ValueError('held-out judging requires a frozen judge manifest (run judge-freeze after calibration)')
    freeze = _load_json(freeze_manifest, {}).get('freeze')
    if not freeze:
        raise ValueError('judge manifest has no freeze record')
    if freeze.get('rubric_hash') != rubric_hash() or freeze.get('schema_hash') != schema_hash():
        raise ValueError('rubric or schema changed after freeze; this held-out set is no longer valid for this configuration')
    if cli_version_now is not None and cli_version_now != freeze.get('cli_version'):
        raise ValueError(f"Claude Code changed since freeze ({freeze.get('cli_version')} -> {cli_version_now}); "
                         'held-out trials must not mix configurations')


def freeze_flags():
    """The judge's CLI flags as recorded in a freeze (prompt and schema values omitted)."""
    return [a for a in build_command('claude', '', {})[1:] if a not in ('', '{}')]


def freeze_judge(manifest_path):
    manifest = _load_json(manifest_path, {})
    probe = manifest.get('probe') or {}
    if not probe.get('passed'):
        raise ValueError('isolation probe must pass before freezing')
    flags = freeze_flags()
    manifest['freeze'] = {'model': MODEL, 'cli_version': probe.get('cli_version'), 'flags': flags,
                          'rubric_hash': rubric_hash(), 'schema_hash': schema_hash(),
                          'probe_hash': probe.get('init_event_hash'),
                          'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    _write_json(manifest_path, manifest)
    return manifest['freeze']


def run_judge(items, out_dir, *, split, trials, caller, ledger, freeze_manifest=None, cli_version_now=None):
    """items: [(run_id, case)]. caller(payload, job_id) -> backend result dict (reserves in ledger itself)."""
    if split == 'heldout':
        _check_freeze(freeze_manifest, cli_version_now)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = _load_json(out / 'key.json', {})
    for run, case in items:
        key[opaque_id(case['case_id'], run)] = {'run': run, 'case_id': case['case_id']}
    _write_json(out / 'key.json', key)
    sent, stopped, failures = 0, None, 0
    for run, case in items:
        oid = opaque_id(case['case_id'], run)
        payload = build_payload(case, oid)
        for trial in range(1, trials + 1):
            job, path = f'{split}:{oid}:t{trial}', out / oid / f't{trial}.json'
            if path.exists() or ledger.done(job):
                continue
            if ledger.status(job) == 'reserved':  # crashed after reserve: recorded as a failed assessment
                ledger.settle(job, 'interrupted', None)
                _write_json(path, {'opaque_id': oid, 'trial': trial, 'split': split, 'status': 'interrupted',
                                   'rubric_hash': rubric_hash(), 'schema_hash': schema_hash()})
                continue
            try:
                result = caller(payload, job)
            except ValueError as error:  # blocked ledger or call cap
                stopped = str(error)
                break
            sent += 1
            if result['status'] in STOP_STATUSES:  # never judged: no record, re-sendable after unblock
                stopped = result['status']
                break
            record = {'opaque_id': oid, 'trial': trial, 'split': split, 'payload_hash': digest(json.dumps(payload, sort_keys=True)),
                      'rubric_hash': rubric_hash(), 'schema_hash': schema_hash(), 'cli_version': result.get('cli_version'),
                      'status': result['status'], 'usage': result.get('usage'), 'attempts': result.get('attempts'),
                      'raw': result.get('raw')}
            if result['status'] == 'complete':
                try:
                    record['result'] = validate_judge_output(payload, result['raw'])
                except JudgeOutputError as error:
                    record.update(status='content_failed', failure_reason=error.reason, failure_detail=str(error))
            _write_json(path, record)
            failures = failures + 1 if record['status'] != 'complete' else 0
            if failures >= MAX_CONSECUTIVE_FAILURES:
                stopped = f'stopped after {failures} consecutive failed assessments; inspect before resuming'
                break
        if stopped:
            break
    return {'split': split, 'sent': sent, 'stopped': stopped, 'calls_used': ledger.calls_used()}


def _overlap(a, b):
    return a['start'] < b['end'] and b['start'] < a['end']


def _merge(lists, span_key, same, value_key):
    merged = [dict(x, trial_support=1) for x in lists[0]]
    for other in lists[1:]:
        unused = list(other)
        for m in merged:
            match = next((o for o in unused if o.get(span_key) and m.get(span_key) and same(m, o)
                          and _overlap(m[span_key], o[span_key])), None)
            if match is None:
                continue
            unused.remove(match)
            m['trial_support'] += 1
            if match[value_key] != m[value_key]:
                m.setdefault('trial_values', [m[value_key]]).append(match[value_key])
                m[value_key] = 'unclear'
                m['trial_disagreement'] = True
        merged += [dict(o, trial_support=1) for o in unused]
    for m in merged:  # found in only some trials: the trials disagree on whether it exists
        if m['trial_support'] < len(lists) and m[value_key] != 'unclear':
            m.setdefault('trial_values', [m[value_key]])
            m[value_key], m['trial_disagreement'] = 'unclear', True
    return merged


def merge_trials(results):
    """Combine validated trials: agreement keeps the label; a differing label, or a claim found
    in only some trials, becomes 'unclear' (spec §6.5)."""
    claims = _merge([r['claims'] for r in results], 'generated_span',
                    lambda a, b: a['subject'] == b['subject'] and a['fact_type'] == b['fact_type'], 'label')
    inventory = _merge([r['section_inventory'] for r in results], 'source_span', lambda a, b: True, 'status')
    return {'trials': len(results), 'claims_complete': all(r['claims_complete'] for r in results),
            'claims': claims, 'section_inventory': inventory}


def load_judge(out_dir):
    """-> (merged {(run, case_id): result}, by_trial {'t1': {...}, 't2': {...}}). Failed trials are left out."""
    out = Path(out_dir)
    key = _load_json(out / 'key.json', {})
    merged, by_trial = {}, {}
    for oid, ident in key.items():
        results = []
        for path in sorted((out / oid).glob('t*.json')):
            record = _load_json(path, {})
            if record.get('status') == 'complete' and 'result' in record:
                results.append(record['result'])
                by_trial.setdefault(path.stem, {})[(ident['run'], ident['case_id'])] = record['result']
        if results:
            merged[(ident['run'], ident['case_id'])] = merge_trials(results)
    return merged, by_trial

