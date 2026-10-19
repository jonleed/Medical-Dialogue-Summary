"""Frozen synthetic capability tests; separate from production/regression accuracy."""
from collections import Counter
from datetime import datetime, timezone
import json
import re
from pathlib import Path

from .checks import CHECKER_VERSION, demographic_suggestions
from .io import file_hash, read_jsonl, write_json, write_jsonl
from .meta import evaluate


def checker_hashes():
    return {name:file_hash(Path(__file__).with_name(name)) for name in ['checks.py','demographics.py']}


def words(n):
    units='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
    tens={20:'twenty',30:'thirty',40:'forty',50:'fifty',60:'sixty',70:'seventy',80:'eighty',90:'ninety'}
    return units[n] if n<20 else tens[n//10*10]+(' '+units[n%10] if n%10 else '')


def _row(family, index, variant, source, output, typ, scope='demographic', visible=None, label='supported', visible_label=None, pair_id=None, edit=None):
    identifier=f'capability:{family}:{index}:{variant}'
    visible=source if visible is None else visible
    target=re.search(r'\b(?:she|he)\b',output,re.I) if typ=='pronoun' else re.search(r'\b\d+\b',output) if typ=='age' else None
    target_span={'start':target.start(),'end':target.end(),'text':target.group()} if target else {'start':0,'end':len(output),'text':output}
    return {'id':identifier,'family':family,'pair_id':pair_id,'variant':variant,'scope':scope,
      'gold_kind':'synthetic','gold_full_source':label,
      'gold_model_visible':visible_label if visible_label is not None else label,
      'target_type':typ,'target_span':target_span,'edit':edit,
      'case':{'case_id':identifier,'dataset_kind':'synthetic','protocol':'capability-controlled-v3',
        'dialogue':source,'prediction':output,'model_input':{'text':visible,'truncated':visible!=source},
        'requested_section':'history of present illness','reference':''}}


def _pair(family,index,source,output,before,after,typ,scope='demographic'):
    if output.count(before)!=1:raise ValueError('Controlled edit must have exactly one location')
    start=output.index(before);end=start+len(before);pair_id=f'{family}:{index}'
    good=_row(family,index,'correct',source,output,typ,scope,pair_id=pair_id)
    bad=_row(family,index,'corrupted',source,output[:start]+after+output[end:],typ,scope,
      label='contradicted',pair_id=pair_id,edit={'start':start,'end':end,'before':before,'after':after})
    return [good,bad]


def build():
    """Generate controlled facts without calling the checker or an LLM for labels."""
    rows=[]
    for index,n in enumerate([28,46,73]):
        w=words(n);wrong=str(n+7);out=f'The patient is a {n}-year-old.'
        age_sources={
          'age_numeric':f'Patient: I am {n} years old.',
          'age_words':f"Patient: I'm {w} years old.",
          'age_qa_self':f"Doctor: How old are you?\nPatient: I'm {w}.",
          'age_qa_bare':f'Doctor: How old are you?\nPatient: {w.capitalize()}.',
          'age_relative_distractor':f'Patient: My father is {n+20} years old. I am {w} years old.',
          'age_historical_distractor':f'Patient: I was twelve when it started. I am {w} years old.',
          'age_birthday':f'Patient: I just turned {w}.',
          'age_family_answer':f'Doctor: Tell me your age.\nGuest_family: She is {w}!',
          'age_confirmed_question':f'Doctor: You are {w} years old?\nPatient: Yes.'}
        for family,source in age_sources.items():rows+=_pair(family,index,source,out,str(n),wrong,'age')
        for family,pred in [('age_aged_syntax',f'The patient, aged {n}, has pain.'),
          ('age_abbreviated_syntax',f'{n} y/o patient with pain.'),('age_header_syntax',f'Age: {n}.')]:
            rows+=_pair(family,index,f'Patient: I am {w} years old.',pred,str(n),wrong,'age')
        pron_sources={
          'pronoun_explicit_patient':'Doctor: The patient is here. She reports pain.',
          'pronoun_self_declared':'Patient: My pronouns are she/her. Please do not refer to me using he or him.',
          'pronoun_companion':'Doctor: Are you the husband?\nGuest_family: I came here with her today.',
          'pronoun_nursing_address':'Doctor: Nurse, please give her a blanket.',
          'pronoun_copular_anchor':'Doctor: She is the patient.',
          'pronoun_family_age_answer':f'Doctor: How old are you?\nGuest_family: She is {w}.'}
        for family,source in pron_sources.items():rows+=_pair(family,index,source,'She reports pain.','She','He','pronoun')
        rows+=_pair('sex_explicit_clinician',index,'Doctor: The patient is female, not male.','The patient is female.','female','male','sex')
        rows+=_pair('sex_explicit_self',index,'Patient: I am female, not male.','The patient is female.','female','male','sex')
        rows+=_pair('gender_explicit_self',index,'Patient: I identify as a woman. I do not identify as a man.','The patient identifies as a woman.','woman','man','gender')
        rows+=_pair('gender_nonbinary_alias',index,'Patient: I identify as nonbinary. I do not identify as a man.','The patient identifies as non-binary.','non-binary','a man','gender')
        clinical=[
          ('medication_dose','Patient: I take aspirin 500 mg daily.','Takes aspirin 500 mg daily.','500','5','quantity'),
          ('medication_identity','Patient: I take aspirin daily. I do not take metformin.','Takes aspirin daily.','aspirin','metformin','medication'),
          ('diagnosis_identity','Doctor: The patient has diabetes and does not have asthma.','The patient has diabetes.','diabetes','asthma','diagnosis'),
          ('symptom_negation','Patient: I have no fever.','No fever.','No','Has','symptom'),
          ('symptom_duration','Patient: Pain started three days ago.','Pain for 3 days.','3','30','quantity'),
          ('followup_date','Doctor: Follow up on March 12.','Follow up on March 12.','March','May','date'),
          ('family_age_quantity','Patient: My father is sixty eight years old.','Father is 68 years old.','68','78','quantity'),
          ('gestational_quantity','Doctor: The patient was born at thirty two weeks.','Born at 32 weeks.','32','30','quantity')]
        # Clinical facts are varied by a neutral case ID, not new independent facts.
        for family,source,pred,before,after,typ in clinical:
            rows+=_pair(family,index,source,pred,before,after,typ,'clinical_scope_probe')
        controls=[
          ('age_unstated','Patient: I have pain.',out,'age','unsupported'),
          ('relative_age_only',f'Patient: My father is {w} years old.',out,'age','unsupported'),
          ('sex_unstated','Patient: I have pain.','The patient is female.','sex','unsupported'),
          ('age_quoted_other',f"Patient: My mother said, 'I am {w} years old.'",out,'age','unsupported'),
          ('address_only','Doctor: Hello, sir.','She reports pain.','pronoun','unclear'),
          ('provider_pronoun_only','Doctor: The patient saw a physician. She recommended rest.','She reports pain.','pronoun','unclear'),
          ('conflicting_age',f'Patient: I am {w} years old.\nPatient: I am {words(n+7)} years old.',out,'age','unclear')]
        for family,source,pred,typ,label in controls:
            rows.append(_row(family,index,'control',source,pred,typ,label=label))
        source=f'Patient: I have pain.\nPatient: I am {w} years old.'
        rows.append(_row('truncated_age',index,'control',source,f'The patient is a {n+7}-year-old.','age',
          visible='Patient: I have pain.',label='contradicted',visible_label='unsupported'))
    # Repeated substitutions must not inflate case counts or leak exact duplicates.
    unique={}
    for row in rows:
        c=row['case'];key=(c['dialogue'],c['prediction'],c['model_input']['text'],row['target_type'])
        unique.setdefault(key,row)
    return list(unique.values())


def freeze(out_dir):
    out=Path(out_dir)
    if out.exists():raise ValueError('Frozen suite already exists')
    rows=build();out.mkdir(parents=True)
    write_jsonl(out/'cases.jsonl',rows)
    manifest={'version':'capability_controlled_v3','created_at':datetime.now(timezone.utc).isoformat(),
      'cases':len(rows),'families':len({r['family'] for r in rows}),
      'gold_kind':'synthetic','exposure':'Implementer-authored prospective challenge, frozen before first scoring; not independent clinical gold or a private holdout.',
      'cases_sha256':file_hash(out/'cases.jsonl'),'builder_sha256':file_hash(__file__),
      'gold_revision_note':'v2 removed duplicates and compound demographic targets and added medication/diagnosis exclusions; v3 additionally supplies explicit identity exclusions so gender/sex categories are not assumed inherently exclusive, and self-declared pronouns explicitly reject the alternative.',
      'checker_version':CHECKER_VERSION,'checker_hashes':checker_hashes(),
      'test_split_accessed':False,'thresholds':{'demographic_contradiction_sensitivity_min':.90,'supported_false_contradiction_max':.05},
      'threshold_note':'Declared engineering targets, not validated clinical safety thresholds.'}
    write_json(out/'manifest.json',manifest)
    return manifest


def run(frozen_dir,out_dir):
    frozen=Path(frozen_dir);out=Path(out_dir)
    if out.exists():raise ValueError('Result output already exists')
    manifest=json.loads((frozen/'manifest.json').read_text())
    if file_hash(frozen/'cases.jsonl')!=manifest['cases_sha256']:raise ValueError('Frozen cases/gold changed')
    if checker_hashes()!=manifest['checker_hashes']:raise ValueError('Checker changed since this suite was frozen; preserve baseline identity')
    rows=read_jsonl(frozen/'cases.jsonl');results=[]
    for row in rows:
        suggestions=[s for s in demographic_suggestions(row['case']) if s['fact_type']==row['target_type'] and
          max(s['generated_span']['start'],row['target_span']['start'])<min(s['generated_span']['end'],row['target_span']['end'])]
        for view in ['full_source','model_visible']:
            labels={s['proposed_label'] if view=='full_source' else s['model_visible_proposed_label'] for s in suggestions}-{None}
            pred=next(iter(labels)) if len(labels)==1 else 'unclear' if labels else 'not_assessed' if suggestions else 'not_extracted'
            results.append({'id':row['id']+':'+view,'gold':row['gold_'+view],'pred':pred,
              'view':view,'extracted':bool(suggestions),'group':row['family'],'scope':row['scope'],'target_type':row['target_type'],
              'pair_id':row['pair_id'],'variant':row['variant'],'case_id':row['id']})
    report={'evaluator_status':'unvalidated_screening_baseline','checker_version':CHECKER_VERSION,
      'checker_hashes':checker_hashes(),'evaluation_code_sha256':{name:file_hash(Path(__file__).with_name(name)) for name in ['challenge.py','meta.py']},'frozen_cases_sha256':manifest['cases_sha256'],
      'gold_kind':'synthetic','cases':len(rows),'family_count':manifest['families'],
      'limitations':manifest['exposure'],'clinical_validation':None,'human_reviews':0}
    for view in ['full_source','model_visible']:
        subset=[r for r in results if r['view']==view]
        report[view]={scope:evaluate([r for r in subset if scope=='all' or r['scope']==scope])
          for scope in ['all','demographic','clinical_scope_probe']}
        report[view]['by_family']={family:evaluate([r for r in subset if r['group']==family],bootstrap=0) for family in sorted({r['group'] for r in subset})}
    primary=report['full_source']['demographic']
    report['engineering_targets_met']={
      'demographic_contradiction_sensitivity':(primary['contradiction_sensitivity']['rate'] or 0)>=manifest['thresholds']['demographic_contradiction_sensitivity_min'],
      'supported_false_contradiction_rate':primary['false_contradiction_rate_on_supported']['rate'] is not None and primary['false_contradiction_rate_on_supported']['rate']<=manifest['thresholds']['supported_false_contradiction_max']}
    out.mkdir(parents=True);write_jsonl(out/'decisions.jsonl',results);write_json(out/'metrics.json',report)
    return report
