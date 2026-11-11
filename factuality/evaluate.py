"""Score any summarizer's outputs on the published benchmark dialogues.

Deterministic parts (validity gate, rule checker, counterfactual outcomes) always run. With a judge
caller, the frozen judge also reviews every valid summary; its rates are reported next to the judge's
measured held-out precision and recall, so a reader knows how far to trust them.
"""
import json
from collections import Counter, defaultdict

from .claude_code_backend import MODEL, require_run_preconditions
from .counterfactual import score_counterfactual
from .io import digest
from .judge import rubric_hash, schema_hash
from .judge_run import freeze_flags, load_judge, run_judge
from .metrics import _overlap
from .rules import suggestions_v3
from .validity import classify_output, validity_summary


def _join(cases, rows):
    joined = []
    for r in rows:
        if r['case_id'] in cases:
            c = dict(cases[r['case_id']], prediction=r.get('prediction'))
            c['model_input'] = r.get('model_input')  # absent -> visibility reported as unclear
            joined.append(c)
    return joined


def _ratio(n, d, **extra):
    return {'numerator': n, 'denominator': d, 'rate': n / d if d else None, **extra}


def _rules(valid):
    sugs = [s for c in valid for s in suggestions_v3(c)]
    by_type = defaultdict(Counter)
    for s in sugs:
        by_type[s['fact_type']][s['proposed_label']] += 1
    return {'suggestions': len(sugs), 'by_type': {k: dict(v) for k, v in sorted(by_type.items())},
            'visibility': dict(Counter(s['visibility'] for s in sugs)),
            'review_required': sum(s['review_required'] for s in sugs)}


def judge_config_differences(bundle, cli_version_now=None):
    """-> (current judge configuration, keys that differ from the published freeze)."""
    freeze = (bundle.judge_config or {}).get('freeze') or {}
    current = {'model': MODEL, 'flags': freeze_flags(), 'rubric_hash': rubric_hash(), 'schema_hash': schema_hash(),
               'cli_version': cli_version_now}
    diffs = [k for k in ('model', 'flags', 'rubric_hash', 'schema_hash', 'cli_version')
             if k in freeze and current[k] is not None and current[k] != freeze[k]]
    return current, diffs


def _check_freeze(bundle, cli_version_now=None, accept_judge_drift=False):
    current, diffs = judge_config_differences(bundle, cli_version_now)
    if diffs and not accept_judge_drift:
        raise ValueError(f"judge configuration changed since the published freeze ({', '.join(diffs)}); the published "
                         'precision/recall would not apply. Use --no-judge, or --accept-judge-drift to run without them.')
    return current, diffs


def judge_preflight(bundle, manifest, cli_version_now, exe=None, accept_judge_drift=False):
    """Freeze check first (so a changed rubric is reported as such), then the local run preconditions."""
    _check_freeze(bundle, cli_version_now, accept_judge_drift)
    if exe is not None:
        require_run_preconditions(manifest, exe)


def _judge_rates(name, valid, merged, plans):
    claims = [c for case in valid for c in (merged.get((name, case['case_id'])) or {}).get('claims', [])]
    assessable = sum(c['label'] in ('supported', 'contradicted', 'unsupported') for c in claims)
    present = omitted = unassessed = 0
    plan_by_case = {p['case_id']: p for p in plans}
    for case in valid:
        result = merged.get((name, case['case_id']))
        if result is None:
            continue
        for fact in (plan_by_case.get(case['case_id']) or {}).get('expected_facts', []):
            if fact.get('label') != 'pending':  # excluded facts never count as omissions
                continue
            item = next((i for i in result.get('section_inventory', [])
                         if any(_overlap(i.get('source_span'), s) for s in fact.get('source_spans', []))), None)
            if item is None or item['status'] not in ('present', 'omitted'):
                unassessed += 1
            elif item['status'] == 'omitted':
                omitted += 1
            else:
                present += 1
    return {'contradicted': _ratio(sum(c['label'] == 'contradicted' for c in claims), assessable),
            'unsupported': _ratio(sum(c['label'] == 'unsupported' for c in claims), assessable),
            'omitted': _ratio(omitted, omitted + present, unassessed=unassessed)}


def _fingerprint(cases):
    rows = [(c['case_id'], c.get('prediction'), c.get('model_input')) for c in sorted(cases, key=lambda c: c['case_id'])]
    return digest(json.dumps(rows, sort_keys=True, ensure_ascii=False))[:10]


def evaluate_model(name, summaries, bundle, *, caller=None, ledger=None, work_dir, reliability=None,
                   cf_summaries=None, cli_version_now=None, accept_judge_drift=False):
    seen = [r['case_id'] for r in summaries]
    duplicates = sorted({i for i in seen if seen.count(i) > 1})
    if duplicates:
        raise ValueError(f'duplicate case_ids in summaries: {duplicates[:5]}')
    ids = set(seen)
    cases = _join(bundle.cases, summaries)
    report = {'model': name,
              'coverage': {'scored': len(cases), 'missing': len(set(bundle.cases) - ids), 'unknown': len(ids - set(bundle.cases))},
              'validity': validity_summary(cases)['counts'] if cases else {}}
    valid = [c for c in cases if classify_output(c.get('prediction'), c['dialogue']) == 'summary']
    report['rules'] = _rules(valid)
    report['counterfactual'] = None
    if cf_summaries:
        report['counterfactual'] = score_counterfactual(_join({c['case_id']: c for c in bundle.cf_cases}, cf_summaries))
    report['judge'] = None
    if caller is not None:
        config, diffs = _check_freeze(bundle, cli_version_now, accept_judge_drift)
        # The judge needs the text the summarizer saw; without it, visibility would be misreported.
        judgeable = [c for c in valid if c.get('model_input')]
        # Keyed on the summaries' content, so changed summaries are never matched to old judge results.
        split = f'eval-{name}-{_fingerprint(judgeable)}'
        summary = run_judge([(name, c) for c in judgeable], work_dir / split, split=split, trials=1, caller=caller,
                            ledger=ledger)
        merged, _ = load_judge(work_dir / split)
        report['judge'] = {'judged': sum((name, c['case_id']) in merged for c in judgeable),
                           'not_judged_no_model_input': len(valid) - len(judgeable), 'stopped': summary['stopped'],
                           'rates': _judge_rates(name, judgeable, merged, bundle.plans),
                           'config': config, 'matches_freeze': not diffs, 'differences': diffs,
                           'reliability': None if diffs else reliability}
    return report
