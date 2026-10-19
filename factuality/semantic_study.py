"""Predeclared empirical study. Hidden physician labels are never sent to the judge."""
import csv
import json
import random
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from .semantic_run import run_batch,save,digest,now,omission_reference,MODEL

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'results/factuality/local/semantic-study-2026-10-18'
PUBLIC=ROOT/'results/factuality/semantic-study-2026-10-18'
MODELS=['biobart','bart-lora','bart-base']

def load_cases():
    return {m:[json.loads(x) for x in (ROOT/f'results/factuality/local/{m}-flask-2026-10-17/cases.jsonl').read_text().splitlines()] for m in MODELS}

def read_result(folder,id):return json.loads((folder/id/'result.json').read_text())

def risk_metrics(pairs):
    n=len(pairs);conf=[[0]*4 for _ in range(4)]
    for y,p in pairs:conf[y-1][p-1]+=1
    fs=[]
    for k in range(4):
        tp=conf[k][k];fp=sum(r[k] for r in conf)-tp;fn=sum(conf[k])-tp
        fs.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    tp=sum(y>=3 and p>=3 for y,p in pairs);fn=sum(y>=3 and p<3 for y,p in pairs)
    fp=sum(y<3 and p>=3 for y,p in pairs);tn=sum(y<3 and p<3 for y,p in pairs)
    return {'n':n,'confusion_gold_rows_prediction_columns_1_to_4':conf,'accuracy':sum(y==p for y,p in pairs)/n if n else None,'macro_f1':sum(fs)/4,'mae':sum(abs(y-p) for y,p in pairs)/n if n else None,'review_required':{'tp':tp,'fn':fn,'fp':fp,'tn':tn,'sensitivity':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,'false_positive_rate':fp/(fp+tn) if fp+tn else None}}

def bootstrap(pairs,seed=20261018,reps=2000):
    # One independent source group per row in this external subset.
    rng=random.Random(seed);values={'accuracy':[],'macro_f1':[],'mae':[],'sensitivity':[],'specificity':[]}
    if not pairs:return {}
    for _ in range(reps):
        m=risk_metrics([rng.choice(pairs) for _ in pairs])
        for key in values:
            v=m.get(key,m['review_required'].get(key))
            if v is not None:values[key].append(v)
    return {k:[sorted(v)[int(.025*(len(v)-1))],sorted(v)[int(.975*(len(v)-1))]] for k,v in values.items() if v}

def external_rows():
    directory=LOCAL/'external/medval'
    manifest=json.loads((directory/'split_manifest.json').read_text())
    content=(directory/'test.csv').read_bytes()
    import hashlib
    if hashlib.sha256(content).hexdigest()!=manifest['public_csv_sha256']:raise ValueError('external dataset changed')
    rows=list(csv.DictReader(content.decode().splitlines(keepends=True)))
    # Preserve embedded dialogue newlines using csv reader.
    selected={r['#']:r for r in rows if r['task']=='dialogue2note'}
    return manifest,selected

def risk_jobs(manifest,rows,split):
    # IDs are opaque ordinals; no dataset ID, reference, label, or assessment enters task.
    return [{'id':'risk-'+str(i).zfill(3),'kind':'risk','task':{'source':rows[r['id']]['input'],'prediction':rows[r['id']]['output'],'requested_section':'complete clinical encounter note'}} for i,r in enumerate(manifest['records']) if r['split']==split]

def score_external(manifest,rows,split,folder):
    pairs=[];omissions=[];errors=[]
    for i,r in enumerate(manifest['records']):
        if r['split']!=split:continue
        result=read_result(folder,'risk-'+str(i).zfill(3))
        if result['status']!='complete':errors.append(r['id']);continue
        row=rows[r['id']];answer=result['result'];pairs.append((int(row['physician_risk_grade']),answer['risk_grade']))
        if omission_reference(row['physician_error_assessment']):
            found=[e for e in answer['errors'] if e['category'] in ['missing_claim','missing_comparison','missing_context']]
            omissions.append({'dataset_row':r['id'],'explicit_physician_omission_positive':True,'judge_found_any_omission':bool(found),'physician_assessment':row['physician_error_assessment'],'judge_omissions':found})
    m=risk_metrics(pairs);m.update({'eligible':sum(r['split']==split for r in manifest['records']),'failures':errors,'source_cluster_bootstrap_95ci':bootstrap(pairs),'omission_positive_cases':len(omissions),'any_omission_detection_on_positive_cases':sum(r['judge_found_any_omission'] for r in omissions)/len(omissions) if omissions else None,'omission_endpoint_limit':'case-level positive agreement; errors may refer to different omitted facts; absence of annotation is not a negative; no omission specificity claim'})
    save(folder/'physician_comparison.json',m);save(folder/'explicit_omission_comparison.json',omissions)
    return m

def run_external():
    if MODEL!='gpt-6-astra':raise ValueError('primary clinical risk protocol requires gpt-6-astra')
    manifest,rows=external_rows();metrics={}
    for effort in ['medium','high']:
        folder=LOCAL/'calibration'/effort
        run_batch(risk_jobs(manifest,rows,'calibration'),folder,effort,workers=6)
        metrics[effort]=score_external(manifest,rows,'calibration',folder)
    if any(m['failures'] for m in metrics.values()):raise ValueError('calibration failures block selection')
    selected=max(['medium','high'],key=lambda e:(metrics[e]['macro_f1'],-metrics[e]['mae'],e=='medium'))
    selection={'selected_effort':selected,'rule':'four-class macro F1, then lower MAE, then medium','calibration':metrics,'frozen_at':now(),'prompt_and_schema_bound_by':json.loads((LOCAL/'calibration'/selected/'batch.json').read_text()),'split_manifest_hash':digest(manifest),'holdout_inspected_for_selection':False}
    selection_path=LOCAL/'selection.json'
    if selection_path.exists():
        old=json.loads(selection_path.read_text())
        if old['selected_effort']!=selected or old['split_manifest_hash']!=digest(manifest):raise ValueError('frozen selection changed')
    else:save(selection_path,selection)
    # Only after persisted selection do held-out predictions/labels enter scoring.
    folder=LOCAL/'holdout'/selected
    run_batch(risk_jobs(manifest,rows,'holdout'),folder,selected,workers=6)
    metrics_holdout=score_external(manifest,rows,'holdout',folder)
    if metrics_holdout['failures']:raise ValueError('held-out inference failures remain; conditional metrics saved locally')
    save(PUBLIC/'clinical_metrics.json',{'published_physician_reference':'stanfordmimi/MedVAL-Bench dialogue2note','revision':manifest['revision'],'calibration':metrics,'selection':{'effort':selected,'frozen_at':json.loads(selection_path.read_text())['frozen_at']},'holdout':metrics_holdout,'limits':['public dataset contamination unknown','published physician grades, not new review of project outputs','only 57 held-out source groups','no physician-validated claim inventory or omission-negative gold']})
    return selected

def run_plans(cases,effort):
    unique={digest([c['dialogue'],c['requested_section']]):c for c in cases['biobart']}
    jobs=[{'id':'plan-'+key[:20],'kind':'plan','task':{'source':c['dialogue'],'requested_section':c['requested_section']}} for key,c in unique.items()]
    folder=LOCAL/'source_plans';run_batch(jobs,folder,effort,workers=6,batch_size=5)
    plans={key:read_result(folder,'plan-'+key[:20]) for key in unique}
    freeze={'frozen_at':now(),'all_complete':all(r['status']=='complete' for r in plans.values()),'hashes':{k:v['request_hash'] for k,v in plans.items()},'result_hashes':{k:digest(v['result']) for k,v in plans.items() if v['status']=='complete'},'requested_models':sorted({v.get('requested_model') for v in plans.values() if v['status']=='complete'}),'kind':'model source-only proposals, not physician gold'}
    freeze_path=folder/'frozen.json'
    if freeze_path.exists():
        prior=json.loads(freeze_path.read_text())
        for key in ['hashes','result_hashes','requested_models']:
            if prior.get(key)!=freeze.get(key):raise ValueError('frozen source-plan results changed')
    else:save(freeze_path,freeze)
    if not all(r['status']=='complete' for r in plans.values()):raise ValueError('incomplete source plans')
    return plans

def run_project(cases,plans,effort):
    jobs=[]
    for model,rows in cases.items():
        for i,c in enumerate(rows):
            key=digest([c['dialogue'],c['requested_section']]);p=plans[key]['result']['facts']
            jobs.append({'id':model+'-'+str(i).zfill(3),'kind':'inventory','task':{'source':c['dialogue'],'prediction':c['prediction'],'requested_section':c['requested_section'],'plans':p}})
    results=run_batch(jobs,LOCAL/'project_inventory',effort,workers=6,batch_size=5)
    if any(r['status']!='complete' for r in results):raise ValueError('project inventory failures remain')

def run_challenge(effort):
    rows=[json.loads(x) for x in (ROOT/'results/factuality/frontier-methods-2026-10-17/challenge-final/cases.jsonl').read_text().splitlines()]
    jobs=[]
    for i,r in enumerate(rows):
        c=r['case'];jobs.append({'id':'challenge-'+str(i).zfill(3),'kind':'inventory','task':{'source':c['dialogue'],'prediction':c['prediction'],'requested_section':c['requested_section'],'plans':[]}})
    results=run_batch(jobs,LOCAL/'challenge_inventory',effort,workers=6,batch_size=5)
    if any(r['status']!='complete' for r in results):raise ValueError('challenge inventory failures remain')

def main():
    PUBLIC.mkdir(parents=True,exist_ok=True)
    phase=sys.argv[1] if len(sys.argv)>1 else 'all'
    if phase=='inventories':
        if MODEL!='gpt-6.1-sol':raise ValueError('inventory protocol requires gpt-6.1-sol')
        cases=load_cases();plans=run_plans(cases,'medium');run_project(cases,plans,'medium');run_challenge('medium')
        save(LOCAL/'inventory_execution_complete.json',{'finished_at':now(),'model':MODEL,'effort':'medium'})
        return
    if phase not in ['all','external']:raise ValueError('phase must be all, external or inventories')
    effort=run_external()
    if phase=='external':return
    env=dict(os.environ,FACTUALITY_JUDGE_MODEL='gpt-6.1-sol')
    subprocess.run([sys.executable,'-m','factuality.semantic_study','inventories'],env=env,cwd=ROOT,check=True)
    save(LOCAL/'execution_complete.json',{'finished_at':now(),'clinical_risk_model':'gpt-6-astra','risk_effort':effort,'inventory_model':'gpt-6.1-sol','inventory_effort':'medium'})

if __name__=='__main__':main()
