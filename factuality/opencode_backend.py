"""Free-only, one-step OpenCode transport with durable reservations."""
import fcntl
import json
import os
import signal
import subprocess
import tempfile
import time
import socket
import secrets
import urllib.request
import base64
from contextlib import contextmanager
from pathlib import Path
from .io import case_hash
from .free_protocol import PROMPT,validate_response

MODEL='opencode/deepseek-v4.1-flash-free'
MAX_ATTEMPTS=24
MAX_OUTPUT=2000
INPUT_BOUND=10000


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


class BudgetLedger:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def _load(self):return json.loads(self.path.read_text()) if self.path.exists() else {'entries':{}}
    def read(self):return self._load()
    def block(self,reason):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=self._load();state['blocked_reason']=reason;save(self.path,state)
    def reserve(self,job_id,model,input_bound):
        if model!=MODEL:raise ValueError('only selected free model is allowed')
        if type(input_bound) is not int or not 0<input_bound<=250000:raise ValueError('invalid input reservation')
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=self._load();entries=state['entries']
            if state.get('blocked_reason'):raise ValueError('provider access blocked: '+state['blocked_reason'])
            if job_id in entries:raise ValueError('job already attempted; no automatic retry')
            if len(entries)>=MAX_ATTEMPTS or any(e['status']=='reserved' for e in entries.values()):raise ValueError('attempt limit or outstanding reservation')
            used_input=sum(e.get('usage',{}).get('input',e['input_bound']) for e in entries.values())
            used_output=sum(e.get('usage',{}).get('output',MAX_OUTPUT) for e in entries.values())
            if used_input+input_bound>250000 or used_output+MAX_OUTPUT>96000:raise ValueError('token allowance exhausted')
            entries[job_id]={'model':model,'status':'reserved','input_bound':input_bound,'output_bound':MAX_OUTPUT,'reserved_at':time.time()};save(self.path,state)
        return job_id
    def settle(self,job_id,usage,status):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=self._load();entry=state['entries'][job_id]
            if entry['status']!='reserved':raise ValueError('reservation already settled')
            if usage is not None:
                if any(type(usage.get(k)) is not int or usage[k]<0 for k in ['input','output']):raise ValueError('invalid actual usage')
                entry['usage']=usage
            entry['status']=status;entry['finished_at']=time.time();save(self.path,state)


def config():
    # Legacy fields are explicitly translated by this installed v2 preview.
    return {'model':MODEL,'small_model':MODEL,'enabled_providers':['opencode'],'default_agent':'factuality','agent':{'factuality':{'mode':'primary','model':MODEL,'prompt':'Evaluate only supplied clinical data. Return JSON. Do not use tools.','maxSteps':1,'permission':{'*':'deny'}},'title':{'disable':True},'summary':{'disable':True},'compaction':{'disable':True}},'compaction':{'auto':False},'warming':False,'snapshots':False,'providers':{'opencode':{'models':{'deepseek-v4.1-flash-free':{'body':{'max_tokens':MAX_OUTPUT,'temperature':0,'thinking':{'type':'disabled'}},'limit':{'context':12000,'input':INPUT_BOUND,'output':MAX_OUTPUT},'settings':{'maxRetries':0}}}}}}


def tools_denied(rules):
    indexes=[i for i,r in enumerate(rules) if r.get('action')=='*' and r.get('resource')=='*']
    return bool(indexes) and all(r.get('effect')=='deny' for r in rules[indexes[-1]:])


@contextmanager
def private_server(executable,cwd):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    password=secrets.token_urlsafe(24)
    env=dict(os.environ,OPENCODE_CONFIG_DIR=str(Path(cwd).resolve()),OPENCODE_SERVER_PASSWORD=password,OPENCODE_PASSWORD=password)
    url='http://127.0.0.1:'+str(port)
    with (Path(cwd)/'private-server.log').open('w') as log:
        process=subprocess.Popen([executable,'serve','--hostname','127.0.0.1','--port',str(port)],cwd=cwd,env=env,stdout=log,stderr=log,start_new_session=True)
        try:
            deadline=time.monotonic()+10
            while True:
                try:
                    auth='Basic '+base64.b64encode(('opencode:'+password).encode()).decode()
                    path='/api/info' if executable=='opencode' else '/api/health'
                    health=json.loads(urllib.request.urlopen(urllib.request.Request(url+path,headers={'Authorization':auth}),timeout=.3).read())
                    if health.get('healthy') or (path=='/api/info' and isinstance(health,dict)):break
                except Exception:
                    if time.monotonic()>deadline or process.poll() is not None:raise ValueError('private OpenCode server did not become healthy')
                    time.sleep(.05)
            def api(path):
                r=subprocess.run([executable,'api','--server',url,'GET',path],cwd=cwd,env=env,capture_output=True,text=True,timeout=30)
                if r.returncode:raise ValueError('OpenCode metadata request failed: '+r.stderr[-300:])
                return json.loads(r.stdout)
            # Agent registration is asynchronous; model catalog waits for plugin initialization.
            api('/api/model')
            agents=api('/api/agent').get('data',[])
            if not any(a['id']=='factuality' and a.get('steps')==1 and tools_denied(a.get('permissions',[])) for a in agents):raise ValueError('restricted reviewer is not registered or permits tools')
            yield url,env,api
        finally:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()

def preflight(executable='opencode'):
    proof={'ready':False,'model':MODEL,'executable':executable,'inference_calls':0,'config_hash':case_hash(config())}
    try:
        with tempfile.TemporaryDirectory(prefix='factuality-opencode-') as tmp:
            save(Path(tmp)/'opencode.json',config())
            version=subprocess.run([executable,'--version'],capture_output=True,text=True,timeout=10,check=True).stdout.strip()
            with private_server(executable,tmp) as (url,env,api):
                docs=api('/api/config');models=api('/api/model');agent_proof=api('/api/agent')
            own=[d['info'] for d in docs if Path(d.get('path','/')).resolve()==(Path(tmp)/'opencode.json').resolve()]
            if not own or len({case_hash(x) for x in own})!=1:raise ValueError('isolated config was not loaded consistently')
            effective=own[-1];agent=effective.get('agents',{}).get('factuality',{})
            if agent.get('steps')!=1 or agent.get('permissions')!=[{'action':'*','resource':'*','effect':'deny'}]:raise ValueError('agent controls not effective')
            if any(not effective.get('agents',{}).get(a,{}).get('disabled') for a in ['title','summary','compaction']):raise ValueError('auxiliary agents not disabled')
            if effective.get('warming') is not False or effective.get('compaction',{}).get('auto') is not False:raise ValueError('background inference not disabled')
            policies=effective.get('policies',effective.get('experimental',{}).get('policies',[]))
            if policies!=[{'action':'provider.use','resource':'*','effect':'deny'},{'action':'provider.use','resource':'opencode','effect':'allow'}]:raise ValueError('provider allowlist missing')
            if any(d.get('info',{}).get(k) for d in docs for k in ['plugins','mcp','instructions','references','skills']):raise ValueError('external extensions/instructions discovered')
            items=models.get('data',[]) if isinstance(models,dict) else models
            if isinstance(items,dict):items=list(items.values())
            chosen=[m for m in items if m.get('providerID')=='opencode' and m.get('modelID')=='deepseek-v4.1-flash-free']
            if len(chosen)!=1:raise ValueError('selected free model metadata missing')
            model=chosen[0]
            if not model.get('enabled'):raise ValueError('selected model is not enabled')
            if model.get('body',{}).get('max_tokens')!=MAX_OUTPUT or model.get('limit',{}).get('input')!=INPUT_BOUND:raise ValueError('model input/output controls not effective')
            for price in model.get('cost',[]):
                if price.get('input',0)!=0 or price.get('output',0)!=0:raise ValueError('selected model is not priced free')
            proof.update(ready=True,transport='opencode-client-private-server',version=version,policy_proof=effective,model_proof={k:model.get(k) for k in ['id','providerID','modelID','enabled','limit','body','cost']},agent_proof=[a for a in agent_proof['data'] if a['id']=='factuality'],provider_weight_revision=None)
    except Exception as exc:proof['blocked_reason']=str(exc)
    return proof


def run_process(command,cwd,timeout=120,env=None):
    process=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True,env=env if env is not None else dict(os.environ,OPENCODE_CONFIG_DIR=str(Path(cwd).resolve())))
    try:
        stdout,stderr=process.communicate(timeout=timeout)
    except BaseException:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.communicate()
        raise
    return process.returncode,stdout,stderr


def parse_events(stdout):
    events=[json.loads(line) for line in stdout.splitlines() if line.strip().startswith('{')]
    texts=[];usage=None;models=set();steps=0
    for event in events:
        part=event.get('part',{});kind=event.get('type')
        if kind in ['tool_use','tool','tool-start'] or part.get('type')=='tool':raise ValueError('unexpected tool request')
        if kind=='text':texts.append(part.get('text',event.get('text','')))
        if kind=='step_finish':
            steps+=1;tokens=part.get('tokens',{})
            usage={'input':int(tokens.get('input',0))+int(tokens.get('cache',{}).get('read',0)),'output':int(tokens.get('output',0))+int(tokens.get('reasoning',0))}
        for obj in [event,event.get('info',{}),part]:
            if obj.get('providerID') and obj.get('modelID'):models.add(obj['providerID']+'/'+obj['modelID'])
    if steps!=1:raise ValueError('expected exactly one completed model step')
    if models and models!={MODEL}:raise ValueError('unexpected model route')
    if not usage or usage['input']<=0:raise ValueError('actual token usage unavailable')
    text=''.join(texts).strip()
    if text.startswith('```') and text.endswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
    return json.loads(text),usage,events


def run_review(job,model,ledger,out,executable='opencode'):
    if model!=MODEL:raise ValueError('model substitution refused')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);prompt=PROMPT+json.dumps(job['payload'],ensure_ascii=False)
    identity=case_hash({'payload':job['payload'],'prompt':PROMPT,'model':model,'config':config(),'version':job['version'],'transport':'private-client-v1'})
    record=out/'result.json'
    if record.exists():
        cached=json.loads(record.read_text())
        if cached['request_hash']!=identity:raise ValueError('stale cached request')
        return cached
    if len(prompt.encode())>7500:raise ValueError('full prompt exceeds conservative input allowance; no truncation')
    ledger.reserve(job['id'],model,INPUT_BOUND);started=time.time();usage=None
    save(out/'request.json',{'payload':job['payload'],'prompt':PROMPT,'model':model,'config':config(),'request_hash':identity})
    try:
        with tempfile.TemporaryDirectory(prefix='factuality-review-') as tmp:
            save(Path(tmp)/'opencode.json',config())
            with private_server(executable,tmp) as (url,env,api):
                cmd=[executable,'run','--server',url,'--model',MODEL,'--agent','factuality','--format','json','--title','Targeted factuality pilot',prompt]
                code,stdout,stderr=run_process(cmd,tmp,env=env)
                events=[json.loads(line) for line in stdout.splitlines() if line.strip().startswith('{')]
                session_ids={e['sessionID'] for e in events if e.get('sessionID')}
                (out/'events.jsonl').write_text(stdout);(out/'stderr.txt').write_text(stderr)
                if code==0:
                    raw,usage,events=parse_events(stdout)
                    if len(session_ids)!=1:raise ValueError('expected one owned reviewer session')
                    prefix='/api/experimental/session/' if executable=='opencode' else '/api/session/'
                    exported=api(prefix+next(iter(session_ids))+'/export');save(out/'session-export.json',exported)
                    refs=[]
                    def collect(value):
                        if isinstance(value,dict):
                            if value.get('providerID') and value.get('id'):refs.append(value['providerID']+'/'+value['id'])
                            for child in value.values():collect(child)
                        elif isinstance(value,list):
                            for child in value:collect(child)
                    collect(exported)
                    if not refs or set(refs)!={MODEL}:raise ValueError('session export does not prove selected model route')
        (out/'events.jsonl').write_text(stdout);(out/'stderr.txt').write_text(stderr)
        if code:
            errors=[e.get('error',{}).get('message','') for e in events if e.get('type')=='error']
            raise ValueError('OpenCode failed: '+'; '.join(errors)+stderr[-300:])
        raw,usage,events=parse_events(stdout)
        if usage['input']>INPUT_BOUND or usage['output']>MAX_OUTPUT:raise ValueError('observed token bound violation')
        result={'id':job['id'],'request_hash':identity,'status':'complete','model':MODEL,'usage':usage,'transport':'opencode-client-private-server','result':validate_response(job['payload'],raw)}
    except Exception as exc:
        result={'id':job['id'],'request_hash':identity,'status':'failed','model':MODEL,'usage':usage,'error':str(exc)}
    finally:
        # Interrupted/unknown usage remains reserved at its full allowance.
        if 'result' in locals():
            ledger.settle(job['id'],usage,result['status'])
            if result['status']=='failed' and any(word in result.get('error','').lower() for word in ['free tier','provider.auth','quota','usage limit','authentication']):ledger.block(result['error'])
    result['elapsed_seconds']=time.time()-started;save(record,result);return result
