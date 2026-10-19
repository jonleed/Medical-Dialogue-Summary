"""Evaluator meta-evaluation. Missing extraction remains an error/coverage gap."""
from collections import Counter, defaultdict
import random
import math

LABELS=('supported','contradicted','unsupported','unclear')
MISSING=('not_extracted','not_graded','not_assessed')


def rate(n,d):
    return {'numerator':n,'denominator':d,'rate':n/d if d else None}


def _metrics(rows):
    correct=sum(r['gold']==r['pred'] for r in rows)
    positive=[r for r in rows if r['gold']=='contradicted']
    supported=[r for r in rows if r['gold']=='supported']
    errors=[r for r in rows if r['gold'] in {'contradicted','unsupported'}]
    return {'accuracy':rate(correct,len(rows)),
      'contradiction_sensitivity':rate(sum(r['pred']=='contradicted' for r in positive),len(positive)),
      'false_contradiction_rate_on_supported':rate(sum(r['pred']=='contradicted' for r in supported),len(supported)),
      'supported_sent_to_review':rate(sum(r['pred'] in {'contradicted','unsupported','unclear'} for r in supported),len(supported)),
      'error_review_capture':rate(sum(r['pred'] in {'contradicted','unsupported','unclear'} for r in errors),len(errors)),
      'extraction_coverage':rate(sum(r.get('extracted',r['pred'] not in MISSING) for r in rows),len(rows)),
      'label_coverage':rate(sum(r['pred'] not in MISSING for r in rows),len(rows)),
      'decisive_coverage':rate(sum(r['pred'] in LABELS[:3] for r in rows),len(rows))}


def evaluate(rows, bootstrap=1000, seed=42):
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate evaluation item IDs')
    if any(r['gold'] not in LABELS or r['pred'] not in LABELS+MISSING or not r.get('group') for r in rows):
        raise ValueError('Unknown label/prediction or missing cluster ID')
    if type(bootstrap) is not int or bootstrap<0:raise ValueError('Invalid bootstrap count')
    result=_metrics(rows)
    result['items']=len(rows)
    result['confusion']={g:{p:sum(r['gold']==g and r['pred']==p for r in rows) for p in LABELS+MISSING} for g in LABELS}
    per_label={}
    for label in LABELS:
        tp=sum(r['gold']==label and r['pred']==label for r in rows)
        actual=sum(r['gold']==label for r in rows);predicted=sum(r['pred']==label for r in rows)
        per_label[label]={'precision':rate(tp,predicted),'recall':rate(tp,actual),
          'f1':2*tp/(actual+predicted) if actual+predicted else None,'support':actual}
    result['per_label']=per_label
    f1s=[v['f1'] for v in per_label.values() if v['support']]
    result['macro_f1']=sum(f1s)/len(f1s) if f1s else None
    result['macro_f1_definition']='Unweighted mean over labels present in gold; missing predictions stay in class recall denominators.'
    pairs=defaultdict(list)
    for row in rows:
        if row.get('pair_id'):pairs[row['pair_id']].append(row)
    complete=[p for p in pairs.values() if len(p)==2 and {r.get('variant') for r in p}=={'correct','corrupted'}]
    result['matched_pairs_both_correct']=rate(sum(all(r['gold']==r['pred'] for r in p) for p in complete),len(complete))
    result['incomplete_pairs']=len(pairs)-len(complete)
    groups=defaultdict(list)
    for row in rows:groups[row['group']].append(row)
    names=sorted(groups);rng=random.Random(seed)
    intervals={k:[] for k in ['accuracy','contradiction_sensitivity','false_contradiction_rate_on_supported']}
    for _ in range(bootstrap if names else 0):
        sample=[r for name in rng.choices(names,k=len(names)) for r in groups[name]]
        metrics=_metrics(sample)
        for key,values in intervals.items():
            if metrics[key]['rate'] is not None:values.append(metrics[key]['rate'])
    result['cluster_bootstrap_95']={'groups':len(names),'resamples':bootstrap,'seed':seed,
      'interpretation':'Resample whole supplied clusters. Synthetic-family intervals describe this designed suite, not a clinical population. Zero observed errors can yield a degenerate [0,0] interval; this does not bound real-world risk.'}
    for key,values in intervals.items():
        values.sort()
        result['cluster_bootstrap_95'][key]=[values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]] if values else None
    supported_groups=[g for g in groups.values() if any(r['gold']=='supported' for r in g)]
    failed_groups=sum(any(r['gold']=='supported' and r['pred']=='contradicted' for r in g) for g in supported_groups)
    n=len(supported_groups)
    if n:
        p=failed_groups/n;z=1.959963984540054;den=1+z*z/n
        center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        result['supported_family_false_alarm_wilson_95']={'families_with_false_alarm':failed_groups,'eligible_families':n,'interval':[max(0,center-half),min(1,center+half)],'note':'Binary any-false-alarm per supplied family; not a clinical population risk estimate.'}
    else:result['supported_family_false_alarm_wilson_95']=None
    return result
