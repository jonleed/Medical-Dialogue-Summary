"""Selected agent-inspected examples, explicitly incomplete and non-random.

MTS-Dialog, Ben Abacha et al., EACL 2023, CC BY 4.0. Source phrases retained;
summary labels/atomization are derived agent additions, not human gold.
"""
from .io import case_hash
from .review import blank_annotation

# Case ID, proposition, type, label, generated evidence, original source evidence.
INSPECTED = [
 ('val:84','Abdominal pain reported','symptom','supported','Abdominal pain','pain in my stomach'),
 ('val:84','Diarrhea reported','symptom','supported','diarrhea','I have had diarrhea'),
 ('val:84','Vomiting reported','symptom','supported','vomiting','Patient: Yes.  I threw up this morning.'),
 ('val:57','No medications reported','medication','supported','None.','Doctor: Are you taking any medications? \nPatient: No.'),
 ('val:72','Hypertension reported','diagnosis','supported','hypertension','I have high blood pressure'),
 ('val:72','Depression reported','diagnosis','supported','depression','I have high blood pressure and, um, depression.'),
 ('val:72','Osteoporosis reported','diagnosis','supported','osteoporosis','Patient: Um, yes, I have osteoporosis too.'),
 ('val:5','No jaundice','symptom','contradicted','he had no jaundice',"He had jaundice but they didn't give any treatment for it."),
 ('val:0','Prior visit on February 2, 2009','date','contradicted','02/02/2009','January tenth two thousand nine'),
 ('val:0','Follow-up on February 3, 2009','date','contradicted','02/03/2009','February tenth two thousand nine'),
 ('val:36','Creatinine normal','other','contradicted','normal creatinine','You do have abnormal levels for Sodium and Creatinine which are related to your kidneys.'),
 ('val:36','Sodium normal','other','contradicted','normal sodium','You do have abnormal levels for Sodium and Creatinine which are related to your kidneys.'),
 ('val:36','Thyroid levels normal','other','contradicted','normal thyroid','Then your thyroid levels are not normal as well.'),
 ('val:47','Flagyl dose 5 mg','quantity','contradicted','Flagyl 5 mg','Let me add Flagyl intra venously. It will be five hundred M G every eight hours'),
 ('val:41','Urolithiasis diagnosed','diagnosis','unsupported','Urolithiasis',None),
]


def evidence(text, phrase):
    start=text.index(phrase)
    return {'start':start,'end':start+len(phrase),'text':phrase}


def selected_examples(cases, reviewed_at):
    by_id={c['case_id']:c for c in cases}
    annotations={}
    excerpts=[]
    for case_id,text,typ,label,generated,source in INSPECTED:
        c=by_id[case_id]
        a=annotations.setdefault(case_id,blank_annotation(c))
        a.update(reviewer='Codex agent inspection',reviewer_kind='agent',status='partial',
          reviewed_at=reviewed_at,unresolved=['Selected propositions only; complete atomic review and blind expected-fact inventory pending.'])
        claim={'id':f"c{len(a['claims'])+1}", 'text':text,'fact_type':typ,'label':label,
          'generated_span':evidence(c['prediction'],generated),
          'source_spans':[evidence(c['dialogue'],source)] if source else [],
          'visible_spans':[evidence(c['model_input']['text'],source)] if source else [],'visibility':'visible',
          'rationale':('Agent inspected the full original source and full visible input: urinary frequency and evaluation for infection are discussed, but no stone diagnosis or urolithiasis is stated. The generated diagnosis is unsupported. Not human gold.' if source is None else 'Agent inspected original and model-visible evidence. Explicit source statement supports or contradicts this selected proposition. Not human gold.')}
        a['claims'].append(claim)
        excerpts.append({'case_id':case_id,'case_hash':case_hash(c),
          'source_csv_sha256':c['source']['csv_sha256'],'dialogue_sha256':c['source']['dialogue_sha256'],
          'requested_section':c['requested_section'],'claim':claim,
          'annotation_status':'partial selected agent inspection; pending human review'})
    return list(annotations.values()), {'attribution':'MTS-Dialog, Ben Abacha et al., EACL 2023, https://github.com/abachaa/MTS-Dialog',
      'source_license':'CC BY 4.0; see MTS-Dialog/LICENSE.txt',
      'modifications':'Selected evidence excerpts, generated claims and agent annotations added.',
      'selection':f'{len(annotations)} selected cases after viewing predictions; not a random reviewed sample. No omission labels.',
      'human_reviewed_cases':0, 'agent_partially_annotated_cases':len(annotations),
      'agent_claims':len(excerpts),'examples':excerpts}
