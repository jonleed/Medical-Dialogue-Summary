"""Evaluate all saved summaries with the local demographic checker. No API calls.

Usage:
    .venv/bin/python -m factuality.evaluate_saved_summaries

Reads the 300 saved cases (100 per model) from results/factuality/local/ and
produces aggregate demographic-screening results under results/factuality/.
"""
import json
import time
from pathlib import Path

from .checks import demographic_suggestions, automatic_summary
from .io import case_hash

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'results' / 'factuality' / 'local'
PUBLIC = ROOT / 'results' / 'factuality'

RUNS = [
    ('biobart-flask-2026-10-17', 'BioBART-large with LoRA'),
    ('bart-lora-flask-2026-10-17', 'BART-base with LoRA'),
    ('bart-base-flask-2026-10-17', 'BART-base, no adapter'),
]


def load_cases(run_dir):
    path = LOCAL / run_dir / 'cases.jsonl'
    with open(path) as f:
        return [json.loads(line) for line in f]


def evaluate_run(run_dir, model_label):
    """Run demographic checks on a single model's 100 saved summaries."""
    cases = load_cases(run_dir)
    all_suggestions = []
    per_case = []

    for case in cases:
        sugs = demographic_suggestions(case)
        all_suggestions.extend(sugs)
        per_case.append({
            'case_id': case['case_id'],
            'section': case.get('requested_section', ''),
            'suggestions': len(sugs),
            'contradictions': sum(s['proposed_label'] == 'contradicted' for s in sugs),
            'supported': sum(s['proposed_label'] == 'supported' for s in sugs),
            'unclear': sum(s['proposed_label'] == 'unclear' for s in sugs),
            'flagged': any(s['review_required'] for s in sugs),
            'types': sorted({s['fact_type'] for s in sugs}),
        })

    by_type = {}
    for typ in ['age', 'pronoun', 'sex', 'gender']:
        subset = [s for s in all_suggestions if s['fact_type'] == typ]
        if subset:
            by_type[typ] = {
                'suggestions': len(subset),
                'contradictions': sum(s['proposed_label'] == 'contradicted' for s in subset),
                'supported': sum(s['proposed_label'] == 'supported' for s in subset),
                'unclear': sum(s['proposed_label'] == 'unclear' for s in subset),
                'model_visible_contradictions': sum(
                    s['model_visible_proposed_label'] == 'contradicted' for s in subset
                ),
            }

    return {
        'run': run_dir,
        'model': model_label,
        'cases': len(cases),
        'automatic': automatic_summary(all_suggestions),
        'by_type': by_type,
        'flagged_cases': sum(c['flagged'] for c in per_case),
        'contradiction_cases': sum(c['contradictions'] > 0 for c in per_case),
        'per_case_summary': per_case,
    }


def cross_model_summary(results):
    """Produce a model-comparison table as structured data."""
    rows = []
    for r in results:
        auto = r['automatic']
        rows.append({
            'model': r['model'],
            'cases': r['cases'],
            'suggestions': auto['suggestions'],
            'contradictions': auto['contradiction_suggestions'],
            'supported': auto['supported_suggestions'],
            'unclear': auto['unclear_suggestions'],
            'flagged_cases': auto['flagged_cases'],
            'contradiction_cases': auto['contradiction_flagged_cases'],
            'model_visible_contradictions': auto['model_visible_contradictions'],
        })
    return rows


def main():
    results = []
    for run_dir, label in RUNS:
        print(f'Evaluating {label} ({run_dir})...')
        result = evaluate_run(run_dir, label)
        results.append(result)
        auto = result['automatic']
        print(f'  {result["cases"]} cases, {auto["suggestions"]} suggestions, '
              f'{auto["contradiction_suggestions"]} contradictions, '
              f'{auto["flagged_cases"]} flagged')

    comparison = cross_model_summary(results)

    # Write public aggregates (no clinical text).
    out_dir = PUBLIC / 'saved-summary-screening-2026-10-18'
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'description': 'Local demographic screening of 300 saved clinical summaries',
        'checker_version': results[0]['automatic']['checker_version'],
        'evaluated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'total_cases': sum(r['cases'] for r in results),
        'models': [{k: v for k, v in r.items() if k != 'per_case_summary'}
                   for r in results],
        'cross_model_comparison': comparison,
        'limits': [
            'Automatic rule-based suggestions, not annotated error rates',
            'Checker precision and recall are unestablished on this sample',
            'Unclear suggestions are review candidates, not confirmed errors',
            'Does not cover unsupported claims or omissions (requires LLM reviewer)',
        ],
    }

    with open(out_dir / 'screening_results.json', 'w') as f:
        json.dump(output, f, indent=2)
        f.write('\n')

    print(f'\nResults written to {out_dir / "screening_results.json"}')
    print(f'\nCross-model comparison:')
    print(f'{"Model":<28} {"Cases":>5} {"Sugg":>5} {"Contra":>6} {"Supp":>5} '
          f'{"Unclear":>7} {"Flagged":>7} {"Vis.Contra":>10}')
    print('-' * 85)
    for row in comparison:
        print(f'{row["model"]:<28} {row["cases"]:>5} {row["suggestions"]:>5} '
              f'{row["contradictions"]:>6} {row["supported"]:>5} '
              f'{row["unclear"]:>7} {row["flagged_cases"]:>7} '
              f'{row["model_visible_contradictions"]:>10}')


if __name__ == '__main__':
    main()
