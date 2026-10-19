"""Provider-neutral blinded judging. File exports make no external model calls."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from .io import case_hash, digest, file_hash, read_jsonl, write_json, write_jsonl
from .meta import LABELS, evaluate
from .scoring import TYPES, validate_annotation, validate_span

RUBRIC_PATH=Path(__file__).with_name('prompts')/'claim_judge_v1.txt'


def claim_task(private_id,source,prediction,fact_type,view,claim_span=None,kind='claim'):
    if view not in {'full_source','model_visible'}:raise ValueError('Unknown source view')
    if kind not in {'claim','summary_inventory'}:raise ValueError('Unknown task kind')
    span=claim_span or {'start':0,'end':len(prediction),'text':prediction}
    if kind=='claim':validate_span(prediction,span)
    payload={'kind':kind,'view':view,'source':source,'prediction':prediction,
      'claim':{'generated_span':span,'fact_type':fact_type} if kind=='claim' else None,
      'rubric':RUBRIC_PATH.read_text(encoding='utf-8')}
    payload['task_id']=digest(str(private_id)+'\n'+view+'\n'+kind+'\n'+str(span))[:24]
    payload['task_hash']=digest(json.dumps(payload,sort_keys=True,ensure_ascii=False))
    return payload


def validate_response(task,response):
    if response.get('task_id')!=task['task_id'] or response.get('task_hash')!=task['task_hash']:
        raise ValueError('Stale or mismatched judge task')
    canonical={k:v for k,v in task.items() if k!='task_hash'}
    if digest(json.dumps(canonical,sort_keys=True,ensure_ascii=False))!=task['task_hash']:raise ValueError('Task content/hash changed')
    if type(response.get('trial')) is not int or response['trial']<0:raise ValueError('Trial must be a nonnegative integer')
    reviewer=response.get('reviewer',{})
    if reviewer.get('kind') not in {'model','human'} or not str(reviewer.get('id','')).strip():raise ValueError('Actual judge identity/kind required')
    if reviewer['kind']=='model' and (not reviewer.get('model_revision') or not reviewer.get('model_family') or not isinstance(reviewer.get('parameters'),dict)):
        raise ValueError('Exact model revision, family and parameters required')
    claims=[response] if task['kind']=='claim' else response.get('claims')
    if not isinstance(claims,list):raise ValueError('Claims must be a list')
    if task['kind']=='summary_inventory':
        if type(response.get('claims_complete')) is not bool:raise ValueError('Claim inventory completeness required')
        if response['claims_complete'] and task['prediction'].strip() and not claims:raise ValueError('Empty inventory cannot complete a nonempty summary')
    seen=set()
    for claim in claims:
        if claim.get('label') not in LABELS or not str(claim.get('rationale','')).strip():raise ValueError('Valid claim label/rationale required')
        spans=claim.get('evidence')
        if not isinstance(spans,list):raise ValueError('Evidence list required')
        if claim['label'] in {'supported','contradicted'} and not spans:raise ValueError('Positive source evidence required')
        for span in spans:validate_span(task['source'],span)
        if task['kind']=='summary_inventory':
            validate_span(task['prediction'],claim.get('generated_span'))
            if claim.get('fact_type') not in TYPES:raise ValueError('Invalid claim type')
            key=(claim['generated_span']['start'],claim['generated_span']['end'],claim['fact_type'])
            if key in seen:raise ValueError('Duplicate inventory claim')
            seen.add(key)
    return response


def _identity(reviewer):
    # Renaming the same model does not create a second judge. Repeats are trials.
    return (reviewer['kind'],reviewer['model_revision'] if reviewer['kind']=='model' else reviewer['id'])


def _independence(reviewer):
    return (reviewer['kind'],reviewer['model_family'] if reviewer['kind']=='model' else reviewer['id'])


def judge_report(tasks,responses,gold=None):
    by_id={t['task_id']:t for t in tasks}
    if len(by_id)!=len(tasks):raise ValueError('Duplicate task IDs')
    seen=set();by_judge=defaultdict(dict);votes=defaultdict(lambda:defaultdict(list));inventories=[]
    reviewer_metadata={};independence={}
    for response in responses:
        if response.get('task_id') not in by_id:raise ValueError('Unknown task ID')
        task=by_id[response['task_id']];validate_response(task,response)
        identity=_identity(response['reviewer']);key=(task['task_id'],identity,response['trial'])
        if key in seen:raise ValueError('Duplicate judge/task/trial; aliases cannot inflate consensus')
        if identity in independence and independence[identity]!=_independence(response['reviewer']):raise ValueError('Conflicting model-family identity')
        config_key=(identity,response['trial'])
        previous=reviewer_metadata.get(config_key)
        if previous:
            fields=['kind','model_revision','model_family','parameters'] if identity[0]=='model' else ['kind','id']
            if any(previous.get(f)!=response['reviewer'].get(f) for f in fields):raise ValueError('Judge config changed within trial')
        seen.add(key);reviewer_metadata[config_key]=response['reviewer'];independence[identity]=_independence(response['reviewer'])
        if task['kind']=='summary_inventory':inventories.append(response);continue
        by_judge[(identity,response['trial'])][task['task_id']]=response
        votes[task['task_id']][identity].append(response['label'])
    gold=gold or []
    gold_ids=[g['task_id'] for g in gold]
    if len(set(gold_ids))!=len(gold_ids):raise ValueError('Duplicate reference task labels')
    for g in gold:
        if g['task_id'] not in by_id or by_id[g['task_id']]['kind']!='claim' or g.get('gold_kind') not in {'synthetic','agent','human'} or g.get('label') not in LABELS or not g.get('group'):
            raise ValueError('Invalid reference label/provenance')
        if g.get('task_hash')!=by_id[g['task_id']]['task_hash']:raise ValueError('Stale reference task label')
    reports=[]
    for (identity,trial),decisions in sorted(by_judge.items()):
        strata={}
        for kind in sorted({g['gold_kind'] for g in gold}):
            strata[kind]={}
            for view in ['full_source','model_visible']:
                subset=[g for g in gold if g['gold_kind']==kind and by_id[g['task_id']]['view']==view]
                strata[kind][view]=evaluate([{'id':g['task_id'],'gold':g['label'],
                  'pred':decisions.get(g['task_id'],{}).get('label','not_graded'),'group':g['group']} for g in subset])
        reports.append({'reviewer':reviewer_metadata[(identity,trial)],'trial':trial,'graded_claim_tasks':len(decisions),
          'by_reference_kind_and_view':strata})
    all_identity=set(identity for identity,_trial in by_judge)
    consensus=[];disagreements=[];pending=[]
    for task in tasks:
        if task['kind']!='claim':continue
        decisions=votes[task['task_id']]
        labels=[label for trials in decisions.values() for label in trials]
        families={independence[identity] for identity in decisions}
        if len(set(labels))>1:
            disagreements.append({'task_id':task['task_id'],'labels':dict(Counter(labels)),'reason':'judge or repeat disagreement'})
        elif labels and len(families)>=2 and set(decisions)==all_identity:
            consensus.append({'task_id':task['task_id'],'label':labels[0],'status':'unadjudicated_jury_proposal'})
        else:pending.append(task['task_id'])
    return {'tasks':len(tasks),'responses':len(responses),'judges':reports,
      'human_calibrated_accuracy':None,
      'human_reference_claims':sum(g['gold_kind']=='human' for g in gold),
      'human_reference_agreement_available':bool(any(g['gold_kind']=='human' for g in gold) and reports),
      'calibration_note':'Reference strata retain declared provenance. Agent/synthetic agreement is not clinician validation. Human calibration and independent held-out validation require actual expert references.',
      'jury':{'consensus_tasks':len(consensus),'disagreement_tasks':len(disagreements),'pending_claim_tasks':len(pending),
        'consensus':consensus,'disagreements':disagreements,'pending':pending,
        'rule':'Unanimous observed trials, all observed claim judges present, at least two declared model families/human identities. Agreement is not proof of correctness.'},
      'summary_inventory_responses':len(inventories),
      'complete_inventory_attestations':sum(r['claims_complete'] for r in inventories),
      'inventory_note':'Inventories require claim alignment and independent completeness review; no automatic whole-summary consensus or omission score.'}


def prepare(cases_paths,out_dir,annotations_paths=None,challenge_dir=None):
    out=Path(out_dir)
    if out.exists():raise ValueError('Judge bundle already exists')
    tasks=[];mapping=[];gold=[];source_plans={}
    annotations_paths=annotations_paths or []
    if annotations_paths and len(annotations_paths)!=len(cases_paths):raise ValueError('One annotation file per case run required')
    for run_index,path in enumerate(cases_paths):
        cases=read_jsonl(path);by_id={c['case_id']:c for c in cases}
        if len(by_id)!=len(cases):raise ValueError('Duplicate case IDs')
        annotations=read_jsonl(annotations_paths[run_index]) if annotations_paths else []
        for annotation in annotations:validate_annotation(by_id[annotation['case_id']],annotation)
        for case in cases:
            if not isinstance(case.get('prediction'),str):raise ValueError('Cannot judge missing generation')
            plan_id=digest(case['dialogue']+'\n'+case['requested_section'])[:24]
            source_plans.setdefault(plan_id,{'plan_id':plan_id,'dialogue':case['dialogue'],'requested_section':case['requested_section'],'expected_facts':[],'expected_facts_complete':False,'frozen':False,'reviewer':None,'status':'pending'})
            for view in ['full_source','model_visible']:
                source=case['dialogue'] if view=='full_source' else (case.get('model_input') or {}).get('text','')
                task=claim_task(f'{run_index}:{case["case_id"]}',source,case['prediction'] or '',None,view,kind='summary_inventory')
                tasks.append(task);mapping.append({'task_id':task['task_id'],'case_file':str(path),'case_id':case['case_id'],'case_hash':case_hash(case),'view':view})
        for annotation in annotations:
            case=by_id[annotation['case_id']]
            for claim in annotation['claims']:
                for view in ['full_source','model_visible']:
                    source=case['dialogue'] if view=='full_source' else (case.get('model_input') or {}).get('text','')
                    task=claim_task(f'{run_index}:{case["case_id"]}:{claim["id"]}',source,case['prediction'],claim['fact_type'],view,claim['generated_span'])
                    tasks.append(task)
                    mapping.append({'task_id':task['task_id'],'case_file':str(path),'case_id':case['case_id'],'case_hash':case_hash(case),'claim_id':claim['id'],'view':view})
                    # A full-source label is only a visible reference when its evidence was reviewed as visible.
                    if view=='full_source' or claim['visibility']=='visible':
                        gold.append({'task_id':task['task_id'],'task_hash':task['task_hash'],'label':claim['label'],
                          'gold_kind':annotation['reviewer_kind'],'group':case['source']['dialogue_id']})
    if challenge_dir:
        frozen=Path(challenge_dir);manifest=json.loads((frozen/'manifest.json').read_text())
        if manifest['cases_sha256']!=file_hash(frozen/'cases.jsonl'):raise ValueError('Challenge hash mismatch')
        for row in read_jsonl(frozen/'cases.jsonl'):
            case=row['case']
            for view in ['full_source','model_visible']:
                source=case['dialogue'] if view=='full_source' else case['model_input']['text']
                task=claim_task(row['id'],source,case['prediction'],row['target_type'],view,row['target_span'])
                tasks.append(task);mapping.append({'task_id':task['task_id'],'case_file':str(frozen/'cases.jsonl'),'case_id':row['id'],'case_hash':case_hash(case),'view':view})
                gold.append({'task_id':task['task_id'],'task_hash':task['task_hash'],
                  'gold_kind':'synthetic','label':row['gold_'+view],'group':row['family']})
    random.Random(20261017).shuffle(tasks)
    out.mkdir(parents=True)
    write_jsonl(out/'tasks.jsonl',tasks);write_jsonl(out/'private_mapping.jsonl',mapping);write_jsonl(out/'private_reference_labels.jsonl',gold)
    write_jsonl(out/'responses.jsonl',[])
    write_jsonl(out/'source_only_plans.jsonl',source_plans.values())
    packets=[]
    for slot,seed in [('reviewer-A',31),('reviewer-B',97)]:
        for view in ['model_visible','full_source']:
            for kind in ['summary_inventory','claim']:
                packet=out/slot/view/kind;packet.mkdir(parents=True)
                ordered=[t for t in tasks if t['view']==view and t['kind']==kind]
                random.Random(seed).shuffle(ordered)
                write_jsonl(packet/'tasks.jsonl',ordered);write_jsonl(packet/'responses.jsonl',[])
                packets.append({'path':str(packet.relative_to(out)),'tasks':len(ordered),'sha256':file_hash(packet/'tasks.jsonl')})
    manifest={'task_count':len(tasks),'summary_inventory_tasks':sum(t['kind']=='summary_inventory' for t in tasks),
      'claim_tasks':sum(t['kind']=='claim' for t in tasks),'reference_labels':len(gold),
      'source_only_plans':len(source_plans),'export_code_sha256':{name:file_hash(Path(__file__).with_name(name)) for name in ['judging.py','meta.py']},'tasks_sha256':file_hash(out/'tasks.jsonl'),'rubric_sha256':file_hash(RUBRIC_PATH),
      'source_files_sha256':{str(p):file_hash(p) for p in list(cases_paths)+list(annotations_paths)},
      'packets':packets,
      'blinding_order':'Independent visible-only reviewers must freeze judgments before any full-source task or source-only plan exposure. Inventory review must precede known-claim calibration for a case. Model judges require fresh isolated context per task. Root tasks.jsonl is an administrative import index, not a reviewer packet.',
      'status':'All tasks pending. No model calls or human review performed. Share tasks only; keep private mapping/reference files from judges.'}
    write_json(out/'manifest.json',manifest);write_json(out/'pending_report.json',judge_report(tasks,[],gold))
    return manifest
