"""v3 rule screening of saved summaries, gated by output validity. No API calls.

Public aggregates contain counts only. Per-case rows (labels/issues, no text) are
written locally because they drive enrichment sampling; the annotator must not read them.
"""
from collections import Counter, defaultdict

from .rules import automatic_summary_v3, suggestions_v3
from .sections import canonical_section
from .validity import LABELS, classify_output

FIELDS = ('fact_type', 'proposed_label', 'issue', 'model_visible_proposed_label', 'visibility', 'review_required')


def per_case_local(runs):
    rows = []
    for run_id, label, cases in runs:
        for case in cases:
            validity = classify_output(case.get('prediction'), case.get('dialogue', ''))
            sugs = suggestions_v3(case) if validity == 'summary' else []
            rows.append({'run': run_id, 'model': label, 'case_id': case['case_id'], 'validity': validity,
                         'canonical_section': canonical_section(case),
                         'suggestions': [{k: s[k] for k in FIELDS} for s in sugs]})
    return rows


def _counts(sugs):
    return {'suggestions': len(sugs), **Counter(s['proposed_label'] for s in sugs),
            'review_required': sum(s['review_required'] for s in sugs)}


def screen_runs(runs):
    rows = per_case_local(runs)
    models = []
    for run_id, label, cases in runs:
        mine = [r for r in rows if r['run'] == run_id]
        by_case = {c['case_id']: c for c in cases}
        sugs = [s for r in mine if r['validity'] == 'summary' for s in suggestions_v3(by_case[r['case_id']])]
        by_type, by_section = defaultdict(list), defaultdict(list)
        for r in mine:
            if r['validity'] != 'summary':
                continue
            by_section[r['canonical_section']].extend(r['suggestions'])
            for s in r['suggestions']:
                by_type[s['fact_type']].append(s)
        auto = automatic_summary_v3(sugs)
        models.append({'run': run_id, 'model': label, 'cases': len(cases),
                       'validity_counts': {v: sum(r['validity'] == v for r in mine) for v in LABELS},
                       'valid_cases': sum(r['validity'] == 'summary' for r in mine),
                       'automatic': auto,
                       'by_type': {k: _counts(v) for k, v in sorted(by_type.items())},
                       'by_section': {k: _counts(v) for k, v in sorted(by_section.items())}})
    return {'checker_version': models[0]['automatic']['checker_version'] if models else None,
            'models': models,
            'limits': ['Automatic rule suggestions, not annotated error rates',
                       'Checker precision/recall are measured separately against reference labels (see metrics.json)',
                       'Non-summary outputs (validity gate) are excluded from all suggestion counts']}
