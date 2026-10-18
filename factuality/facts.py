"""Finite local extraction; records are proposals, never exhaustive clinical gold."""
import re
from .demographics import age_evidence,age_claims,pronoun_evidence,pronoun_claims,identity_evidence,identity_claims
from .checks import demographic_suggestions
from .io import case_hash


def span(text,quote):
    if not isinstance(quote,str) or not quote:raise ValueError('nonempty exact quote required')
    start=text.find(quote)
    if start<0:raise ValueError('evidence quotation absent from text')
    return {'start':start,'end':start+len(quote),'text':quote}


def extract_facts(text,*,dialogue):
    records=[]
    for typ,extractor in [('age',age_evidence if dialogue else age_claims),('pronoun',pronoun_evidence if dialogue else pronoun_claims),('sex',lambda t:identity_evidence(t,'sex') if dialogue else identity_claims(t,'sex')),('gender',lambda t:identity_evidence(t,'gender') if dialogue else identity_claims(t,'gender'))]:
        for fact in extractor(text):
            records.append({'subject':'patient','fact_type':typ,'value':fact['value'],'unit':'years' if typ=='age' else None,'negated':False,'certainty':'uncertain' if fact['uncertain'] else 'explicit','temporal_context':None,'span':fact['span'],'provenance':fact['basis']})
    patterns=[('quantity',r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|grams?|weeks?|days?)\b'),('medication',r'\b(?:takes?|taking)\s+([A-Za-z][A-Za-z-]+)'),('symptom',r'\b(?:denies|no|without)\s+(fever|cough|pain|nausea|vomiting)\b')]
    for typ,pattern in patterns:
        for m in re.finditer(pattern,text,re.I):
            clause=re.split(r'[.!?\n]',text[:m.start()])[-1]
            relative=bool(re.search(r'\b(father|mother|brother|sister|husband|wife)\b',clause,re.I))
            quantity=re.match(r'(\d+(?:\.\d+)?)\s*(.*)',m.group()) if typ=='quantity' else None
            records.append({'subject':'relative' if relative else 'unresolved','fact_type':typ,'value':float(quantity[1]) if quantity else m.group(1).lower(),'unit':quantity[2].lower() if quantity else None,'negated':typ=='symptom' or bool(re.search(r'\b(not|never|no)\b',clause,re.I)),'certainty':'candidate','temporal_context':None,'span':{'start':m.start(),'end':m.end(),'text':m.group()},'provenance':'explicit_pattern_v1'})
    return records


def freeze_expected(dialogue,section,reference,candidates):
    ids=set();facts=[]
    for c in candidates:
        if not c['id'] or c['id'] in ids:raise ValueError('duplicate/missing source fact ID')
        if type(c.get('required')) is not bool:raise ValueError('required must be boolean')
        ids.add(c['id']);evidence=span(dialogue,c['quote'])
        eligible=c.get('section',section)==section
        facts.append(dict(c,source_span=evidence,required=bool(c['required']) and eligible,eligibility='in_section' if eligible else 'out_of_section',provenance=c.get('provenance','source_fact_proposal')))
    plan={'dialogue':dialogue,'section':section,'reference':reference,'facts':facts,'exhaustive':False}
    return dict(plan,hash=case_hash(plan))


def compare_demographics(case):return demographic_suggestions(case)
