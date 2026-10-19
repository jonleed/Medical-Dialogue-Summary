"""Reproducible automatic demographic audit and optional development regressions."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .checks import automatic_summary, demographic_suggestions
from .io import case_hash, file_hash, read_jsonl, write_json, write_jsonl
from .scoring import validate_annotation


def compare_reviewed(cases, annotations, suggestions):
    """Compare only selected reviewed propositions, never claim overall recall.

    Pronouns are deduplicated per case/category by the checker, so a reviewed
    'his' can match the first patient 'he'. Other dimensions require overlapping
    generated spans. Matching never reads the reference or source label to
    produce automatic suggestions.
    """
    by_id={c['case_id']:c for c in cases}
    rows=[]
    for annotation in annotations:
        validate_annotation(by_id[annotation['case_id']],annotation)
        for claim in annotation['claims']:
            if claim['fact_type'] not in {'age','pronoun','sex','gender'}:continue
            matches=[]
            for s in suggestions:
                if s['case_id']!=annotation['case_id'] or s['fact_type']!=claim['fact_type']:continue
                a,b=s['generated_span'],claim['generated_span']
                overlap=max(a['start'],b['start'])<min(a['end'],b['end'])
                word=b['text'].lower()
                category='masculine' if word in {'he','him','his'} else 'feminine' if word in {'she','her','hers'} else None
                if overlap or (claim['fact_type']=='pronoun' and category==s['generated_value']):matches.append(s)
            rows.append({'case_id':annotation['case_id'],'claim_id':claim['id'],
              'reviewer_kind':annotation['reviewer_kind'],'fact_type':claim['fact_type'],
              'reviewed_label':claim['label'],'generated_text':claim['generated_span']['text'],
              'automatic_labels':sorted({s['proposed_label'] for s in matches}),
              'review_flag_captured':any(s['review_required'] for s in matches),
              'contradiction_captured':any(s['proposed_label']=='contradicted' for s in matches),
              'automatic_supported':any(s['proposed_label']=='supported' for s in matches)})
    contrad=[r for r in rows if r['reviewed_label']=='contradicted']
    other=[r for r in rows if r['reviewed_label'] in {'unsupported','unclear'}]
    supported=[r for r in rows if r['reviewed_label']=='supported']
    return {'purpose':'Development regressions on selected existing annotations used to improve the rules; not independent validation or comprehensive precision/recall.',
      'matching':'Same case and dimension; overlapping generated spans, or the same deduplicated pronoun category.',
      'reviewer_kinds':dict(Counter(r['reviewer_kind'] for r in rows)),
      'selected_claims':len(rows),
      'known_contradictions':len(contrad),'known_contradictions_captured':sum(r['contradiction_captured'] for r in contrad),
      'unsupported_or_unclear_claims':len(other),'unsupported_or_unclear_flagged_for_review':sum(r['review_flag_captured'] for r in other),
      'supported_claims':len(supported),'supported_claims_automatically_supported':sum(r['automatic_supported'] for r in supported),
      'supported_claims_falsely_contradicted':sum(r['contradiction_captured'] for r in supported),
      'comparisons':rows}


def audit_run(cases_path, out_dir, annotations_path=None):
    cases=read_jsonl(cases_path)
    if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('Duplicate case IDs')
    annotations=read_jsonl(annotations_path) if annotations_path else []
    suggestions=[dict(s,case_hash=case_hash(c)) for c in cases for s in demographic_suggestions(c)]
    regressions=compare_reviewed(cases,annotations,suggestions) if annotations_path else None
    out=Path(out_dir)
    if out.exists():raise ValueError('Audit output already exists; use a new path')
    out.mkdir(parents=True)
    summary={'created_at':datetime.now(timezone.utc).isoformat(),'cases':len(cases),
      'cases_file':str(cases_path),'cases_sha256':file_hash(cases_path),
      'annotations_file':str(annotations_path) if annotations_path else None,
      'annotations_sha256':file_hash(annotations_path) if annotations_path else None,
      'code_sha256':{p.name:file_hash(p) for p in [Path(__file__).with_name('demographics.py'),Path(__file__).with_name('checks.py'),Path(__file__)]},
      'automatic':automatic_summary(suggestions),
      'by_type_and_label':dict(Counter(s['fact_type']+'/'+s['proposed_label'] for s in suggestions)),
      'reviewed_regressions':regressions}
    write_jsonl(out/'automatic_suggestions.jsonl',suggestions)
    write_json(out/'summary.json',summary)
    return summary
