"""Single-model targeted pilot; preparation/reporting never invokes inference."""
import argparse
import json
import time
from pathlib import Path
from .free_pilot_cases import build_cases
from .free_protocol import make_payload,PROMPT
from .opencode_backend import MODEL,config,save,preflight,BudgetLedger,run_review
from .free_report import build_report,write_report
from .io import case_hash


def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);cases=build_cases()
    identity=case_hash({'cases':cases,'prompt':PROMPT,'model':MODEL,'config':config()})
    manifest={'freeze_hash':identity,'cases':24,'model':MODEL,'max_attempts':24,'frozen_at':time.time(),'gold_kind':'synthetic construction; not clinical holdout'}
    path=out/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())['freeze_hash']!=identity:raise ValueError('different pilot already frozen')
        return json.loads(path.read_text())
    save(out/'cases.json',cases);save(out/'prompt.json',{'text':PROMPT});save(path,manifest)
    return manifest


def load_frozen(out):
    manifest=json.loads((out/'manifest.json').read_text());cases=json.loads((out/'cases.json').read_text())
    identity=case_hash({'cases':cases,'prompt':PROMPT,'model':MODEL,'config':config()})
    if identity!=manifest['freeze_hash']:raise ValueError('frozen cases/prompt/config/model changed')
    return manifest,cases


def run(out,limit=24):
    manifest,cases=load_frozen(out);ledger=BudgetLedger(out/'ledger.json')
    if ledger.read().get('blocked_reason'):
        save(out/'execution.json',{'status':'blocked','reason':ledger.read()['blocked_reason'],'attempts':len(ledger.read()['entries']),'production_started':False});return
    proof=preflight();save(out/'preflight.json',proof)
    if not proof['ready']:
        save(out/'execution.json',{'status':'blocked','reason':proof['blocked_reason'],'attempts':len(BudgetLedger(out/'ledger.json').read()['entries'])});return
    ledger=BudgetLedger(out/'ledger.json');dispatched=0;status='complete';reason=None
    for case in cases:
        path=out/'reviews'/case['id']/'result.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result['status']!='complete':continue  # Retain its missing assessment; never retry it.
            # Re-enter the pure cache path to verify current request/config identity.
            run_review({'id':case['id'],'payload':make_payload(case),'version':proof['version']},MODEL,ledger,path.parent)
            continue
        if dispatched>=limit:status='paused_at_limit';break
        result=run_review({'id':case['id'],'payload':make_payload(case),'version':proof['version']},MODEL,ledger,path.parent);dispatched+=1
        print(json.dumps({'case':case['id'],'status':result['status'],'attempts':len(ledger.read()['entries']),'usage':result.get('usage')}),flush=True)
        if result['status']!='complete':
            status='stopped';reason=result['error']
            if any(word in reason.lower() for word in ['free tier','provider.auth','quota','usage limit','authentication']):ledger.block(reason)
            break
    if status=='complete' and any(json.loads(p.read_text())['status']!='complete' for p in (out/'reviews').glob('*/result.json')):status='partial'
    save(out/'execution.json',{'status':status,'reason':reason,'attempts':len(ledger.read()['entries']),'finished_at':time.time(),'production_started':False})


def report(out):
    _,cases=load_frozen(out);responses=[json.loads(p.read_text()) for p in sorted((out/'reviews').glob('*/result.json'))]
    result=build_report(cases,responses,BudgetLedger(out/'ledger.json').read());write_report(result,out,cases,responses)
    public=Path(__file__).resolve().parents[1]/'results/factuality/opencode-pilot-2026-10-18'
    save(public/'metrics.json',{k:v for k,v in result.items() if k!='decisions'})
    save(public/'execution.json',json.loads((out/'execution.json').read_text()) if (out/'execution.json').exists() else {'status':'prepared','attempts':0})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','preflight','run','report']);p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int,default=24)
    a=p.parse_args()
    if not 0<a.limit<=24:p.error('limit must be between 1 and 24')
    if a.stage=='prepare':print(prepare(a.out))
    elif a.stage=='preflight':
        proof=preflight();save(a.out/'preflight.json',proof);print({k:v for k,v in proof.items() if k not in ['policy_proof','model_proof']})
    elif a.stage=='run':run(a.out,a.limit)
    else:print(json.dumps(report(a.out),indent=2))

if __name__=='__main__':main()
