"""Blind planning and local, escaped review formats. No automatic reference labels."""
import csv
import html
from pathlib import Path

from .checks import CHECKER_VERSION, demographic_suggestions, automatic_summary
from .io import case_hash, source_plan_hash, write_jsonl


def blank_annotation(case):
    return {'case_id':case['case_id'], 'case_hash':case_hash(case), 'reviewer':'',
      'reviewer_kind':'human', 'status':'pending', 'claims_complete':False,
      'expected_facts_complete':False, 'expected_facts_frozen':False,
      'expected_facts_blind':False, 'expected_facts_reviewer_kind':None, 'expected_facts_reviewer':'', 'expected_facts_frozen_at':None,
      'claims':[], 'expected_facts':[], 'unresolved':['Source-only expected facts and atomic claim review pending.']}


def blind_plans(cases):
    return [{'case_id':c['case_id'], 'source_plan_hash':source_plan_hash(c), 'dialogue':c['dialogue'],
      'requested_section':c['requested_section'], 'reference':c['reference'],
      'expected_facts':[], 'expected_facts_complete':False, 'expected_facts_frozen':False,
      'expected_facts_blind':False, 'expected_facts_reviewer_kind':None, 'expected_facts_reviewer':'', 'expected_facts_frozen_at':None}
      for c in cases]


def write_queue(cases, out_dir, plans=None):
    out = Path(out_dir)
    if out.exists(): raise ValueError('Review output directory already exists; use a new path')
    out.mkdir(parents=True)
    annotations = [blank_annotation(c) for c in cases]
    if plans is not None:
        by_id = {p['case_id']:p for p in plans}
        if len(by_id) != len(plans) or set(by_id) != {c['case_id'] for c in cases}:
            raise ValueError('Plan case IDs must match the complete case set exactly')
        for a in annotations:
            plan = by_id[a['case_id']]
            if plan['source_plan_hash'] != source_plan_hash(next(c for c in cases if c['case_id']==a['case_id'])): raise ValueError('Stale source-only plan')
            for field in ['expected_facts','expected_facts_complete','expected_facts_frozen',
                          'expected_facts_blind','expected_facts_reviewer_kind','expected_facts_reviewer','expected_facts_frozen_at']:
                a[field] = plan[field]
            if a['expected_facts'] or a['expected_facts_complete']:
                a['status'] = 'partial'
                a['reviewer'] = plan['expected_facts_reviewer']
                a['reviewer_kind'] = plan['expected_facts_reviewer_kind']
    write_jsonl(out/'annotations.jsonl', annotations)
    suggestions = [s for c in cases for s in demographic_suggestions(c)]
    write_jsonl(out/'automatic_suggestions.jsonl', suggestions)
    with open(out/'review.csv', 'x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id','status','section','dialogue','model_visible_input','prediction','reference','suggestions'])
        writer.writeheader()
        for c in cases:
            row = {'case_id':c['case_id'], 'status':'pending', 'section':c['requested_section'],
              'dialogue':c['dialogue'], 'model_visible_input':(c.get('model_input') or {}).get('text',''),
              'prediction':c.get('prediction') or '', 'reference':c['reference'],
              'suggestions':str([s for s in suggestions if s['case_id']==c['case_id']])}
            writer.writerow({k:('\t'+v if v.startswith(('=','+','-','@')) else v) for k,v in row.items()})
    esc = lambda text: html.escape(str(text))
    cards = []
    for c in cases:
        flags=[s for s in suggestions if s['case_id']==c['case_id'] and s['review_required']]
        flag_html=''.join('<li>'+esc(s['fact_type']+' / '+s['proposed_label']+' / '+s['issue'])+
          '<br>Generated: '+esc(s['generated_span']['text'])+
          '<br>Full-source evidence: '+esc(' | '.join(x['text'] for x in s['source_spans']) or 'Not established')+
          '<br>Evidence visibility: '+esc(s['visibility'])+
          '<br>Model-visible comparison: '+esc(s['model_visible_proposed_label'])+
          '<br>'+esc(s['rationale'])+'</li>' for s in flags)
        cards.append(f'<details><summary>{esc(c["case_id"])} — {esc(c["requested_section"])} — pending — {len(flags)} automatic review flags</summary>' +
          '<h3>Automatic suggestions (pending)</h3><ul>'+flag_html+'</ul>' +
          ''.join(f'<h3>{title}</h3><pre>{esc(text)}</pre>' for title,text in [
            ('Original source',c['dialogue']), ('Model-visible evidence',(c.get('model_input') or {}).get('text','Unavailable')),
            ('Prediction',c.get('prediction') or 'Generation pending'), ('Reference (coverage aid)',c['reference'])])+'</details>')
    (out/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Factuality review queue</title>'
      '<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #aaa;padding:1rem;margin:1rem 0}summary{cursor:pointer}</style>'
      '<h1>Pending factuality review</h1><p>Complete and freeze the separate source-only plan before opening any predictions. '
      f'Checker: {CHECKER_VERSION}. Suggestions are automatic, not reference labels. Edit atomic annotations in annotations.jsonl; score validates evidence spans and hashes.</p>'
      + ''.join(cards)+'</html>', encoding='utf-8')
    return {'cases':len(cases), **automatic_summary(suggestions), 'human_reviewed_cases':0}
