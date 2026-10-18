"""One pinned free API reviewer; no agents, tools, paid fallback, or retries."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path
from .opencode_backend import BudgetLedger, save, MAX_OUTPUT, INPUT_BOUND
from .free_protocol import PROMPT, make_payload, validate_response
from .free_pilot import load_frozen
from .free_report import build_report, write_report, binary_metrics
from .io import case_hash

MODEL = 'nvidia/nemotron-3-ultra-550b-a55b:free'
KEY_FILE = Path.home()/'.config/medical-factuality/openrouter.env'
ORIGINAL = Path(__file__).resolve().parents[1]/'results/factuality/local/opencode-pilot-2026-10-18'
OUT = ORIGINAL.parent/'openrouter-pilot-2026-10-18'
# Predeclared development subset fits the 19 attempts remaining in the original cap.
SELECTED = [1,3,4,6,7,8,9,10,11,12,13,14,16,19,20,21,22,23,24]

class RouterLedger(BudgetLedger):
    allowed_model = MODEL

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('API redirect refused')


def credentials(path):
    path=Path(path)
    if path.stat().st_uid!=os.getuid() or path.stat().st_mode & 0o077:
        raise ValueError('key file must be owned by this user and private (chmod 600)')
    values={}
    for line in path.read_text().splitlines():
        if '=' in line:
            name,value=line.split('=',1);values[name.strip()]=value.strip()
    if not values.get('OPENROUTER_API_KEY'):raise ValueError('local API key is empty')
    if values.get('OPENROUTER_MODEL')!=MODEL:raise ValueError('key-file model differs from the pinned free model')
    return values['OPENROUTER_API_KEY']


def api(path, key=None, body=None):
    headers={'Content-Type':'application/json'}
    if key:headers['Authorization']='Bearer '+key
    request=urllib.request.Request('https://openrouter.ai/api/v1/'+path,headers=headers,data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.build_opener(NoRedirect()).open(request,timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # Do not log request headers, credentials, or an uncontrolled provider body.
        raise ValueError('OpenRouter HTTP '+str(error.code)+'; no retry or paid fallback') from None


def verify_endpoint(data):
    if data.get('id')!=MODEL:raise ValueError('exact free model metadata required')
    endpoints=[e for e in data.get('endpoints',[]) if e.get('tag')=='nvidia' and e.get('model_id')==MODEL]
    if len(endpoints)!=1:raise ValueError('one pinned NVIDIA endpoint required')
    endpoint=endpoints[0];pricing=endpoint.get('pricing',{})
    if not all(k in pricing for k in ['prompt','completion']):raise ValueError('missing price metadata')
    if any(Decimal(str(v))!=0 for k,v in pricing.items() if k!='discount'):raise ValueError('nonzero endpoint price refused')
    if not {'max_tokens','temperature','reasoning'}<=set(endpoint.get('supported_parameters',[])):raise ValueError('required model parameters unavailable')
    return endpoint


def request_body(payload):
    text=PROMPT+json.dumps(payload,ensure_ascii=False)+'\nOUTPUT CONTRACT: expected_facts must contain exactly one item per TASK.plans entry, using its id. If TASK.plans is empty, return expected_facts: []. source_facts and rule_proposals are NOT omission plans; never invent plan IDs.'
    if len(text.encode())>7500:raise ValueError('prompt exceeds conservative input allowance; no truncation')
    return {'model':MODEL,'messages':[{'role':'user','content':text}],
            'temperature':0,'max_tokens':MAX_OUTPUT,'reasoning':{'enabled':False},'stream':False,
            'provider':{'only':['nvidia'],'allow_fallbacks':False,'require_parameters':True,
                        'max_price':{'prompt':0,'completion':0,'request':0}}}


def parse_response(response):
    if response.get('model') not in [MODEL,MODEL.removesuffix(':free')]:raise ValueError('unexpected response model')
    if response.get('provider','').lower()!='nvidia':raise ValueError('unexpected provider')
    usage=response.get('usage',{})
    if usage.get('cost') is None or Decimal(str(usage['cost']))!=0:raise ValueError('zero actual cost not established')
    if any(type(usage.get(k)) is not int or usage[k]<0 for k in ['prompt_tokens','completion_tokens']):raise ValueError('actual tokens unavailable')
    if not 0<usage['prompt_tokens']<=INPUT_BOUND or usage['completion_tokens']>MAX_OUTPUT:raise ValueError('observed token limit violation')
    choices=response.get('choices',[])
    if len(choices)!=1 or choices[0].get('finish_reason')!='stop':raise ValueError('incomplete response or unexpected choices')
    message=choices[0]['message']
    if message.get('tool_calls') or message.get('function_call'):raise ValueError('tool request refused')
    content=message.get('content','').strip()
    if content.startswith('```') and content.endswith('```'):content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
    return json.loads(content),{'input':usage['prompt_tokens'],'output':usage['completion_tokens']}


def validate_completion(payload, raw):
    try:
        return {'status':'complete','result':validate_response(payload,raw)}
    except (ValueError, KeyError, TypeError) as error:
        # Invalid evidence is a measured missing assessment, not a provider outage.
        return {'status':'failed','failure_stage':'evidence_validation','error':str(error)}


def prepare(out):
    manifest,cases=load_frozen(ORIGINAL)
    identity=case_hash({'cases':cases,'prompt':PROMPT,'request':request_body({}),'selected':SELECTED})
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())['freeze_hash']!=identity:raise ValueError('pilot freeze changed')
        return
    old=json.loads((ORIGINAL/'ledger.json').read_text())
    if len(old['entries'])!=5 or any(e['status']=='reserved' for e in old['entries'].values()):raise ValueError('prior attempt accounting changed')
    out.mkdir(parents=True,exist_ok=True)
    save(out/'cases.json',cases)
    save(out/'ledger.json',{'entries':old['entries'],'prior_ledger_hash':case_hash(old),'prior_provider_block':old['blocked_reason'],'authorized_route':'OpenRouter free API after local key setup'})
    save(out/'manifest.json',{'freeze_hash':identity,'source_freeze':manifest['freeze_hash'],'model':MODEL,'selected_case_ids':[f'case-{i:02d}' for i in SELECTED],'max_total_attempts':24,'prior_attempts':5,'new_request_cap':19,'frozen_at':time.time()})


def frozen(out):
    manifest=json.loads((out/'manifest.json').read_text());cases=json.loads((out/'cases.json').read_text())
    identity=case_hash({'cases':cases,'prompt':PROMPT,'request':request_body({}),'selected':SELECTED})
    if identity!=manifest['freeze_hash']:raise ValueError('frozen pilot identity mismatch')
    return manifest,cases


def report(out):
    manifest,cases=frozen(out)
    responses=[json.loads(p.read_text()) for p in sorted((out/'reviews').glob('*/result.json'))]
    metrics=build_report(cases,responses,RouterLedger(out/'ledger.json').read(),model=MODEL)
    metrics['selection']={'planned':19,'total_frozen':24,'reason':'five previous provider failures consume the original 24-attempt cap','selected_case_ids':manifest['selected_case_ids']}
    metrics['assessed_only_by_capability']={cap:binary_metrics([(r['gold_error'],r['prediction']) for r in metrics['decisions'] if r['capability']==cap and r['prediction'] is not None]) for cap in ['age','gender','unsupported','omission']}
    metrics['actual_cost_usd']=sum(r.get('cost_usd',0) for r in responses)
    metrics['cost_accounting_complete']=all('cost_usd' in r for r in responses)
    write_report(metrics,out,cases,responses)
    public=OUT.parents[1]/'openrouter-pilot-2026-10-18'
    save(public/'metrics.json',{k:v for k,v in metrics.items() if k!='decisions'})
    if (out/'execution.json').exists():save(public/'execution.json',json.loads((out/'execution.json').read_text()))
    return metrics


def run(out,key_file,limit):
    manifest,cases=frozen(out);ledger=RouterLedger(out/'ledger.json')
    if ledger.read().get('blocked_reason'):raise ValueError('run stopped: '+ledger.read()['blocked_reason'])
    key=credentials(key_file)
    endpoint=verify_endpoint(api('models/'+MODEL+'/endpoints')['data'])
    account=api('key',key)['data'];remaining=account.get('free_model_daily_requests',{}).get('remaining')
    if type(remaining) is not int or remaining<1:raise ValueError('no verified free requests remaining')
    save(out/'preflight.json',{'model':MODEL,'endpoint':endpoint,'free_requests_remaining':remaining,'paid_fallback':False,'checked_at':time.time()})
    selected=set(manifest['selected_case_ids']);count=0;status='partial';reason=None
    for case in cases:
        if case['id'] not in selected:continue
        target=out/'reviews'/case['id'];body=request_body(make_payload(case));identity=case_hash(body)
        if (target/'result.json').exists():
            cached=json.loads((target/'result.json').read_text())
            if cached['request_hash']!=case_hash(json.loads((target/'request.json').read_text())):raise ValueError('cached request record changed')
            if cached['status']=='complete' and cached['request_hash']!=identity:raise ValueError('cached review identity changed')
            continue
        if count>=min(limit,remaining):break
        job='openrouter:'+case['id'];ledger.reserve(job,MODEL,INPUT_BOUND);count+=1;usage=None
        save(target/'request.json',body)
        result={'id':case['id'],'model':MODEL,'request_hash':identity,'status':'failed','transport':'openrouter-direct-free'}
        try:
            response=api('chat/completions',key,body);save(target/'response.json',response)
            observed=response.get('usage',{})
            if all(type(observed.get(k)) is int and observed[k]>=0 for k in ['prompt_tokens','completion_tokens']):usage={'input':observed['prompt_tokens'],'output':observed['completion_tokens']}
            if observed.get('cost') is not None:result['cost_usd']=float(observed['cost'])
            raw,usage=parse_response(response)
            result.update(validate_completion(make_payload(case),raw))
            result.update(provider=response['provider'],response_model=response['model'])
        except Exception as error:
            reason=str(error).replace(key,'[REDACTED]');result['error']=reason;ledger.block(reason);status='stopped'
        finally:
            result['usage']=usage;ledger.settle(job,usage,result['status']);save(target/'result.json',result)
        save(out/'execution.json',{'status':status,'reason':reason,'model':MODEL,'total_attempts':len(ledger.read()['entries']),'production_started':False,'paid_fallback':False})
        report(out)
        print(json.dumps({'case':case['id'],'status':result['status'],'usage':usage,'cost_usd':result.get('cost_usd'),'error':result.get('error')}),flush=True)
        if reason:break
    if all((out/'reviews'/i/'result.json').exists() for i in selected) and not reason:
        status='planned_subset_processed_with_invalid_reviews' if any(json.loads((out/'reviews'/i/'result.json').read_text())['status']!='complete' for i in selected) else 'planned_subset_complete'
    save(out/'execution.json',{'status':status,'reason':reason,'model':MODEL,'total_attempts':len(ledger.read()['entries']),'production_started':False,'paid_fallback':False})
    report(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','run','report']);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--key-file',type=Path,default=KEY_FILE);p.add_argument('--limit',type=int,default=19)
    args=p.parse_args()
    if not 1<=args.limit<=19:p.error('limit must be 1..19')
    if args.stage=='prepare':prepare(args.out)
    elif args.stage=='run':run(args.out,args.key_file,args.limit)
    else:print(json.dumps(report(args.out),indent=2))

if __name__=='__main__':main()
