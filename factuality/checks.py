"""Versioned demographic suggestions. Automatic output is always pending review."""
from .demographics import (age_evidence, age_claims, pronoun_claims, identity_evidence, identity_claims,
    gender_class, gender_reference_evidence)

CHECKER_VERSION = 'patient_demographics_v2.1'


def _visible_spans(facts, visible):
    spans=[]
    for fact in facts:
        phrase=fact['span']['text']
        position=visible.find(phrase)
        if position>=0:
            spans.append({'start':position,'end':position+len(phrase),'text':phrase})
    return spans


def _suggest(case, typ, claim, facts, visible_facts):
    visible=(case.get('model_input') or {}).get('text','')
    values={f['value'] for f in facts}
    uncertain=any(f['uncertain'] for f in facts)
    issue='demographic_comparison'
    if len(values)>1 or uncertain:
        label='unclear';issue='conflicting_or_uncertain_source'
        rationale='Source patient evidence is conflicting or qualified; do not resolve it automatically.'
    elif len(values)==1:
        label='supported' if claim['value'] in values else 'contradicted'
        rationale='Compared a patient-attributed generated demographic with explicit source evidence of the same dimension.'
    else:
        label='unclear';issue=typ+'_not_established'
        rationale='No reliable same-dimension patient evidence was extracted. This is a review candidate, not proof of an unsupported claim or of a correct identity.'
    spans=_visible_spans(facts,visible)
    same_visible={f['value'] for f in visible_facts}
    if spans:
        visibility='visible'
    elif facts and (case.get('model_input') or {}).get('truncated'):
        visibility='truncated'
    else:visibility='unclear'
    # A source-backed identity elsewhere is not evidence available to the model.
    visible_label=None
    if len(same_visible)==1 and not any(f['uncertain'] for f in visible_facts):
        visible_label='supported' if claim['value'] in same_visible else 'contradicted'
    elif len(same_visible)>1:visible_label='unclear'
    if label=='unclear':visible_label='unclear' if spans else None
    return {'case_id':case['case_id'],'fact_type':typ,'generated_span':claim['span'],
      'source_spans':[f['span'] for f in facts], 'visible_spans':spans,
      'proposed_label':label,'model_visible_proposed_label':visible_label,
      'visibility':visibility,'status':'pending','method':CHECKER_VERSION,
      'issue':issue,'review_required':label!='supported', 'rationale':rationale,
      'generated_value':claim['value'],'source_values':sorted(values),
      'evidence_bases':sorted({f['basis'] for f in facts})}


def gender_reference_suggestion(case, typ, claim):
    """Compare a summary's sex term or pronoun with every gender-reference cue in the dialogue.

    Sex and pronouns are one dimension: a summary saying "male" or "he" is checked against explicit
    statements, patient-referring pronouns and gendered address alike. A claim that matches the cues
    is supported, one that conflicts is contradicted, and cues that disagree with each other (or no
    cue at all) are unclear: absence of a cue is a review candidate, not proof of an unsupported claim.
    """
    visible=(case.get('model_input') or {}).get('text','')
    s=_suggest(case,typ,dict(claim,value=gender_class(claim['value'])),
               gender_reference_evidence(case['dialogue']),gender_reference_evidence(visible))
    if s['issue']==typ+'_not_established':
        s['issue']='gender_reference_not_established'
    s['generated_value']=claim['value']
    return s


def demographic_suggestions(case):
    source=case['dialogue'];output=case.get('prediction') or ''
    visible=(case.get('model_input') or {}).get('text','')
    suggestions=[]
    for typ,extract,claims in [('age',age_evidence,age_claims),('gender',lambda t:identity_evidence(t,'gender'),lambda t:identity_claims(t,'gender'))]:
        facts=extract(source);visible_facts=extract(visible)
        for claim in claims(output):
            suggestions.append(_suggest(case,typ,claim,facts,visible_facts))
    suggestions+=[gender_reference_suggestion(case,'pronoun',c) for c in pronoun_claims(output)]
    suggestions+=[gender_reference_suggestion(case,'sex',c) for c in identity_claims(output,'sex')]
    return suggestions


def automatic_summary(suggestions):
    """Count pending rule outputs independently of any annotated error rates."""
    return {'checker_version':CHECKER_VERSION, 'suggestions':len(suggestions),
      'supported_suggestions':sum(s['proposed_label']=='supported' for s in suggestions),
      'contradiction_suggestions':sum(s['proposed_label']=='contradicted' for s in suggestions),
      'unclear_suggestions':sum(s['proposed_label']=='unclear' for s in suggestions),
      'flagged_cases':len({s['case_id'] for s in suggestions if s['review_required']}),
      'contradiction_flagged_cases':len({s['case_id'] for s in suggestions if s['proposed_label']=='contradicted'}),
      'model_visible_contradictions':sum(s['model_visible_proposed_label']=='contradicted' for s in suggestions),
      'truncated_source_contradictions':sum(s['proposed_label']=='contradicted' and s['visibility']=='truncated' for s in suggestions),
      'status':'unreviewed suggestions, excluded from annotated rates',
      'checker_precision':None, 'checker_recall':None}
