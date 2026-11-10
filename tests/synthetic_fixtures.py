"""Explicitly synthetic known-label diagnostics; never empirical clinical data."""
from factuality.io import case_hash


def span(text, phrase):
    start=text.index(phrase)
    return {'start':start,'end':start+len(phrase),'text':phrase}


def fixtures():
    # name, source, output, label, type, positive evidence, visible input
    rows = [
      ('correct_age','Patient: I am 42 years old.','The patient is 42 years old.','supported','age','I am 42 years old.',None),
      ('wrong_age','Patient: I am 42 years old.','The patient is 24 years old.','contradicted','age','I am 42 years old.',None),
      ('missing_age','Patient: I am 42 years old. I have pain.','The patient reports pain.','supported','symptom','I have pain.',None),
      ('relative_age','Patient: My mother is 42 years old.','The patient has a mother aged 42.','supported','other','My mother is 42 years old.',None),
      ('dose','Patient: I take 42 mg of X.','Takes 42 mg of X.','supported','medication','I take 42 mg of X.',None),
      ('duration','Patient: Pain began 42 years ago.','Pain began 42 years ago.','supported','quantity','Pain began 42 years ago.',None),
      ('explicit_sex_contradiction','Doctor: The patient is female.','The patient is male.','contradicted','sex','The patient is female.',None),
      ('explicit_gender_contradiction','Patient: I identify as a woman.','The patient identifies as a man.','contradicted','gender','I identify as a woman.',None),
      ('ambiguous_identity','Doctor: She is here with her sister.','The patient is female.','unclear','sex','She is here with her sister.',None),
      ('missing_identity','Patient: Hello.','The patient is female.','unsupported','sex',None,None),
      ('invented_medication','Patient: I have pain.','Takes aspirin.','unsupported','medication',None,None),
      ('invented_diagnosis','Patient: I have pain.','Has diabetes.','unsupported','diagnosis',None,None),
      ('negation','Patient: I have pain.','Denies pain.','contradicted','symptom','I have pain.',None),
      ('relevant_omission','Patient: I have pain and nausea.','Reports pain.','supported','symptom','I have pain and nausea.',None),
      ('out_of_section','Patient: I have pain. My mother has diabetes.','Reports pain.','supported','symptom','I have pain.',None),
      ('truncated_omission','Patient: I have pain.\nPatient: I am 42 years old.','Reports pain.','supported','symptom','I have pain.','Patient: I have pain.'),
      ('source_supported_beyond_input','Patient: Hello.\nPatient: I am 42 years old.','The patient is 42 years old.','supported','age','I am 42 years old.','Patient: Hello.'),
    ]
    cases=[]; annotations=[]
    for name,source,output,label,typ,evidence,visible in rows:
        visible=visible if visible is not None else source
        c={'schema_version':1,'dataset_kind':'synthetic','case_id':'synthetic:'+name,
          'protocol':'synthetic-diagnostic','dialogue':source,'prediction':output,
          'requested_section':'chief complaint' if name=='out_of_section' else 'history of present illness',
          'reference':'Synthetic diagnostic only.',
          'model_input':{'text':visible,'truncated':visible!=source}}
        claim={'id':'c1','text':output,'fact_type':typ,'label':label,
          'generated_span':span(output,output), 'source_spans':[span(source,evidence)] if evidence else [],
          'visible_spans':[span(visible,evidence)] if evidence and evidence in visible else [],
          'visibility':'visible' if not evidence or evidence in visible else 'truncated',
          'rationale':'Known synthetic fixture construction; not a human clinical annotation.'}
        a={'case_id':c['case_id'],'case_hash':case_hash(c),'reviewer':'Synthetic fixture specification',
          'reviewer_kind':'synthetic','status':'complete','claims_complete':True,'expected_facts_complete':True,
          'expected_facts_frozen':True,'expected_facts_blind':True,
          'expected_facts_reviewer_kind':'synthetic', 'expected_facts_reviewer':'Synthetic fixture specification',
          'expected_facts_frozen_at':'2026-10-17T00:00:00Z', 'claims':[claim],'expected_facts':[],
          'unresolved':[]}
        if name in {'missing_age','truncated_omission','relevant_omission'}:
            evidence='I am 42 years old.' if 'age' in name or name=='truncated_omission' else 'nausea'
            a['expected_facts'].append({'id':'f1','text':evidence,'fact_type':'age' if 'years old' in evidence else 'symptom',
              'label':'omitted','visibility':'visible' if evidence in visible else 'truncated',
              'source_spans':[span(source,evidence)],'visible_spans':[span(visible,evidence)] if evidence in visible else [],
              'rationale':'Relevant fact predeclared by synthetic construction; omitted from output.'})
        if name=='out_of_section':
            a['expected_facts'].append({'id':'f1','text':'Maternal diabetes','fact_type':'diagnosis','label':'excluded',
              'visibility':'visible','source_spans':[span(source,'My mother has diabetes.')],
              'visible_spans':[span(visible,'My mother has diabetes.')], 'rationale':'Family history excluded for chief complaint diagnostic.'})
        cases.append(c);annotations.append(a)
    return cases,annotations
