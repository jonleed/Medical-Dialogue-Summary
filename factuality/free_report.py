"""Target-level pilot metrics. Unassessed controls never become true negatives."""
import html
import json
from .opencode_backend import save


def binary_metrics(pairs):
    tp=sum(g and p is True for g,p in pairs);fn=sum(g and p is not True for g,p in pairs)
    fp=sum(not g and p is True for g,p in pairs);tn=sum(not g and p is False for g,p in pairs)
    return {'tp':tp,'fn':fn,'fp':fp,'tn':tn,'eligible':len(pairs),'unassessed':sum(p is None for g,p in pairs),'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def assess(case,response):
    gold=case['gold']
    if not response or response.get('status')!='complete':return {'prediction':None,'label':'not_assessed','extracted':False,'correct':False}
    raw=response['result']
    if gold['capability']=='omission':
        facts=[f for f in raw['expected_facts'] if f['id']==gold['expected_fact_id']]
        label=facts[0]['label'] if len(facts)==1 else 'not_assessed';extracted=bool(facts)
    else:
        target=gold['target_span']
        matches=[c for c in raw['claims'] if c['fact_type']==gold['fact_type'] and c['generated_span']['start']<target['end'] and c['generated_span']['end']>target['start']]
        labels={m['label'] for m in matches};label=next(iter(labels)) if len(labels)==1 else 'not_extracted' if not matches else 'unclear';extracted=bool(matches)
    error_label={'age':'contradicted','gender':'contradicted','unsupported':'unsupported','omission':'omitted'}[gold['capability']]
    pred=None if label in ['unclear','not_assessed','not_extracted'] else label==error_label
    return {'prediction':pred,'label':label,'extracted':extracted,'correct':label==gold['label']}


def build_report(cases,responses,ledger,model='opencode/deepseek-v4.1-flash-free'):
    by_id={r['id']:r for r in responses};rows=[]
    for case in cases:
        decision=assess(case,by_id.get(case['id']));rows.append({'id':case['id'],'capability':case['gold']['capability'],'gold_error':case['gold']['error'],'gold_label':case['gold']['label'],**decision})
    entries=ledger['entries'];known=[e['usage'] for e in entries.values() if e.get('usage')]
    return {'model':model,'cases':len(cases),'semantic_performance_status':'measured' if any(r['status']=='complete' for r in responses) else 'unavailable: no valid provider reviews; missing-target counts are coverage failures, not model capability estimates','valid_reviews':sum(r['status']=='complete' for r in responses),'attempts':len(entries),'by_capability':{cap:binary_metrics([(r['gold_error'],r['prediction']) for r in rows if r['capability']==cap]) for cap in ['age','gender','unsupported','omission']},'target_extraction':{'numerator':sum(r['extracted'] for r in rows),'denominator':len(rows)},'end_to_end_correct':{'numerator':sum(r['correct'] for r in rows),'denominator':len(rows)},'usage':{'known_input':sum(u['input'] for u in known),'known_output':sum(u['output'] for u in known),'unknown_attempts':len(entries)-len(known),'reserved_input_upper_bound':sum(e['input_bound'] for e in entries.values()),'reserved_output_upper_bound':sum(e['output_bound'] for e in entries.values()),'note':'failed provider responses did not expose actual token usage; unknown is not zero'},'decisions':rows,'limits':['24 controlled development cases; not independent clinical validation','model proposals are not physician labels','missing targets remain in denominators','provider model alias; weight revision not exposed']}


def write_report(report,out,cases,responses):
    save(out/'metrics.json',report)
    esc=lambda x:html.escape(str(x));by_id={r['id']:r for r in responses};cards=[]
    for case,decision in zip(cases,report['decisions']):
        cards.append('<details><summary>'+esc(case['id']+' '+decision['capability']+' '+decision['label'])+'</summary><pre>'+esc(json.dumps({'source':case['source'],'prediction':case['prediction'],'gold':case['gold'],'response':by_id.get(case['id'])},indent=2))+'</pre></details>')
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Targeted factuality pilot</title><style>body{font:16px system-ui;max-width:1000px;margin:2rem auto}pre{white-space:pre-wrap}details{padding:1rem;border:1px solid #aaa}</style><h1>Targeted factuality pilot</h1><p>Synthetic known targets; model judgments, not physician review.</p>'+''.join(cards))
