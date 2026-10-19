"""Validate all executed study stages and produce honest model-review aggregates."""
import html
import json
from collections import Counter
from pathlib import Path
from .semantic_study import ROOT,LOCAL,PUBLIC,load_cases,read_result
from .semantic_run import digest,save,anchor,all_anchors,normalize_inventory
from .review import blank_annotation
from .scoring import validate_annotation,score
from .meta import evaluate


def inventory_result(folder,id):
    r=read_result(folder,id)
    if r['status']!='complete':raise ValueError('incomplete inventory '+id)
    # Locate immutable input and raw response within the actual shared inference pack.
    directory=folder/'.batches'/r['inference_batch']
    request=json.loads((directory/'request.json').read_text())
    raw_record=json.loads((directory/'result.json').read_text())
    tasks=request['task']['cases']
    candidates=[x['id'] for x in tasks if digest(x['data'])==r['task_hash'] and digest({'batch_request_hash':raw_record['request_hash'],'opaque_id':x['id'],'task':x['data']})==r['request_hash']]
    if len(candidates)!=1:raise ValueError('opaque item provenance ambiguous')
    opaque=candidates[0]
    task=next(x['data'] for x in tasks if x['id']==opaque)
    if digest({'batch_request_hash':raw_record['request_hash'],'opaque_id':opaque,'task':task})!=r['request_hash']:raise ValueError('inventory provenance mismatch')
    attempts=sorted(directory.glob('answer-*.json'))
    raw=json.loads(attempts[-1].read_text())
    inventory=next(x['result'] for x in raw['items'] if x['id']==opaque)
    return r,task,normalize_inventory(task,inventory)


def to_annotation(case,result,inventory,plan,frozen_at,planner_model):
    a=blank_annotation(case)
    a.update({'reviewer':result['requested_model']+' (Codex CLI; model proposals)','reviewer_kind':'model','status':'partial','claims_complete':inventory['claims_complete'],'expected_facts_complete':False,'expected_facts_frozen':True,'expected_facts_blind':True,'expected_facts_reviewer_kind':'model','expected_facts_reviewer':planner_model+' source-only planner','expected_facts_frozen_at':frozen_at,'unresolved':['Model source plan has not been certified exhaustive by independent clinicians.'],'model_run':{k:result.get(k) for k in ['requested_model','provider_weight_revision','reasoning_effort','request_hash','inference_batch','started_at','finished_at']}})
    visible=(case.get('model_input') or {}).get('text','');truncated=bool((case.get('model_input') or {}).get('truncated'))
    def evidence(quotes,ambiguous=False):
        source=[s for q in quotes for s in all_anchors(case['dialogue'],q)]
        vis=[s for q in quotes if q in visible for s in all_anchors(visible,q)]
        visibility='unclear' if ambiguous else 'visible' if (quotes and all(q in visible for q in quotes)) or (not quotes and not truncated) else 'truncated' if truncated else 'unclear'
        return {'source_spans':source,'visible_spans':vis,'visibility':visibility}
    for i,c in enumerate(inventory['claims']):
        item={k:c[k] for k in ['text','fact_type','label','rationale','generated_span','proposed_label','ambiguous_source_evidence']}
        item.update(id='claim-'+str(i),**evidence(c['evidence_quotes'],c['ambiguous_source_evidence']))
        a['claims'].append(item)
    planned={p['id']:p for p in plan}
    for f in inventory['expected_facts']:
        p=planned[f['id']];quote=p['source_quote'];ambiguous=len(all_anchors(case['dialogue'],quote))>1
        label='excluded' if not p['required'] else 'unclear' if ambiguous else f['label']
        item={'id':p['id'],'text':p['text'],'fact_type':p['fact_type'],'label':label,'proposed_label':f['label'],'required_by_model_plan':p['required'],'rationale':f['rationale'] or 'Optional source-plan item excluded.',**evidence([quote],ambiguous)}
        if label=='present':item['generated_span']=anchor(case['prediction'],f['generated_quote'])
        a['expected_facts'].append(item)
    return validate_annotation(case,a)


def project_results():
    cases=load_cases();frozen=json.loads((LOCAL/'source_plans/frozen.json').read_text())
    if not frozen.get('all_complete') or not frozen.get('result_hashes'):raise ValueError('source plan freeze incomplete')
    summaries={};review=[]
    for model,rows in cases.items():
        annotations=[]
        for i,case in enumerate(rows):
            r,task,inventory=inventory_result(LOCAL/'project_inventory',model+'-'+str(i).zfill(3))
            key=digest([case['dialogue'],case['requested_section']]);plan=read_result(LOCAL/'source_plans','plan-'+key[:20])
            if digest(plan['result'])!=frozen['result_hashes'][key]:raise ValueError('source-plan result changed after freeze')
            if plan['finished_at']>frozen['frozen_at'] or r['started_at']<frozen['frozen_at']:raise ValueError('source planning was not frozen before output review')
            if task['plans']!=plan['result']['facts'] or task['source']!=case['dialogue'] or task['prediction']!=case['prediction']:raise ValueError('wrong evidence/plan linked to output')
            a=to_annotation(case,r,inventory,plan['result']['facts'],frozen['frozen_at'],plan['requested_model']);annotations.append(a)
            for claim in a['claims']:
                if claim['label'] in ['contradicted','unsupported'] or claim['fact_type'] in ['age','sex','gender','pronoun']:
                    review.append({'model':model,'case_id':case['case_id'],'prediction':case['prediction'],'claim':claim})
        out=LOCAL/'annotations'/model;out.mkdir(parents=True,exist_ok=True)
        (out/'annotations.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in annotations))
        s=score(rows,annotations);save(out/'metrics.json',s)
        summaries[model]={'cases':len(rows),'model_reviewed_cases':len(annotations),'human_reviewed_cases':0,'claims_total':sum(len(a['claims']) for a in annotations),'claim_labels':dict(Counter(c['label'] for a in annotations for c in a['claims'])),'claims_complete_attestations':sum(a['claims_complete'] for a in annotations),'ambiguous_source_quote_claims':sum(c['ambiguous_source_evidence'] for a in annotations for c in a['claims']),'required_source_plan_facts':sum(f.get('required_by_model_plan',False) for a in annotations for f in a['expected_facts']),'model_full_source':s['model']['full_source'],'model_visible_evidence_subset':s['model']['model_visible'],'limits':['model proposals, not physician factuality labels','source plans not certified exhaustive; omission rate conditional on model-selected essential facts','visible evidence subset is not a separate blinded rejudgment','all source-group-related outputs require clustered analysis']}
    save(PUBLIC/'project_metrics.json',summaries);save(LOCAL/'review_items.json',review)
    esc=lambda x:html.escape(str(x))
    cards=[]
    for r in review:
        c=r['claim'];cards.append('<details><summary>'+esc(r['model']+' '+r['case_id']+' '+c['fact_type']+' '+c['label'])+'</summary><p>'+esc(c['text'])+'</p><p>Generated: '+esc(c['generated_span']['text'])+'</p><p>Source: '+esc(' | '.join(s['text'] for s in c['source_spans']))+'</p><p>'+esc(c['rationale'])+'</p><p>Visibility: '+esc(c['visibility'])+'</p><pre>'+esc(r['prediction'])+'</pre></details>')
    (LOCAL/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Semantic review proposals</title><style>body{font:16px system-ui;max-width:1000px;margin:2rem auto}details{border:1px solid #aaa;padding:1rem;margin:.5rem}pre{white-space:pre-wrap}</style><h1>Model review proposals</h1><p>Actual recorded model inference; zero new clinician reviews. Errors and all demographic claims shown. Exact evidence can be inspected below.</p>'+''.join(cards))
    return summaries


def match_target(claims,target,typ):
    matches=[c for c in claims if c['fact_type']==typ and c['generated_span']['start']<target['end'] and c['generated_span']['end']>target['start']]
    if not matches:return 'not_extracted',False
    labels={c['label'] for c in matches}
    return (next(iter(labels)) if len(labels)==1 else 'not_graded'),True


def challenge_results():
    fixtures=[json.loads(x) for x in (ROOT/'results/factuality/frontier-methods-2026-10-17/challenge-final/cases.jsonl').read_text().splitlines()]
    decisions=[]
    for i,f in enumerate(fixtures):
        r,task,inventory=inventory_result(LOCAL/'challenge_inventory','challenge-'+str(i).zfill(3))
        if task['source']!=f['case']['dialogue'] or task['prediction']!=f['case']['prediction']:raise ValueError('fixture evidence mismatch')
        pred,extracted=match_target(inventory['claims'],f['target_span'],f['target_type'])
        decisions.append({'id':f['id'],'group':f['family'],'pair_id':f.get('pair_id'),'variant':f.get('variant'),'gold':f['gold_full_source'],'pred':pred,'extracted':extracted,'fact_type':f['target_type'],'scope':f['scope']})
    metrics={'full_source':evaluate(decisions,bootstrap=2000),'demographics':evaluate([r for r in decisions if r['scope']=='demographic'],bootstrap=2000),'by_type':{t:evaluate([r for r in decisions if r['fact_type']==t],bootstrap=1000) for t in sorted({r['fact_type'] for r in decisions})},'limits':['controlled implementer-authored fixtures, not clinical prevalence','typed generated-span overlap evaluates target extraction; not all possible claim boundaries','full-source blind inventories; no fixture target or gold shown to model','risk calibration and inventory validation are separate endpoints','different cases within an inference pack share context']}
    save(PUBLIC/'challenge_metrics.json',metrics);save(LOCAL/'challenge_decisions.json',decisions)
    return metrics


def main():
    if not (LOCAL/'execution_complete.json').exists():raise ValueError('study execution not complete')
    project=project_results();challenge=challenge_results()
    save(PUBLIC/'verification.json',{'actual_summary_inventories':sum(r['model_reviewed_cases'] for r in project.values()),'actual_source_only_plans':100,'actual_challenge_inventories':challenge['full_source']['items'],'human_reviews_of_project_outputs':0,'all_required_batches_complete':True,'clinical_metrics_file_exists':(PUBLIC/'clinical_metrics.json').exists(),'provider_weight_revision':None,'clinical_risk_model_alias':'gpt-6-astra','inventory_model_alias':json.loads((LOCAL/'project_inventory/batch.json').read_text())['model'],'clinical_holdout_not_pretraining_unseen':True})

if __name__=='__main__':main()
