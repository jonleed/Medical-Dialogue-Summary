"""Paired per-case overlap results; factuality stays pending until annotated."""
import json
from pathlib import Path
from .io import read_jsonl, file_hash
from .rouge import compute_rouge


def compare_runs(run_dirs):
    if len(run_dirs) < 2: raise ValueError('At least two runs are required')
    if len({Path(p).name for p in run_dirs}) != len(run_dirs):
        raise ValueError('Comparison run directory names must be unique')
    runs=[]
    for directory in run_dirs:
        root=Path(directory)
        manifest=json.loads((root/'generation_manifest.json').read_text())
        if manifest['status']!='complete': raise ValueError('Incomplete generation run')
        if file_hash(root/'cases.jsonl')!=manifest['cases_sha256']: raise ValueError('Run cases hash mismatch')
        cases=read_jsonl(root/'cases.jsonl')
        if len({c['case_id'] for c in cases})!=len(cases): raise ValueError('Duplicate paired case IDs')
        runs.append((root.name,manifest,cases,compute_rouge(cases)))
    first=runs[0]
    def evidence(cases):
        return [(c['case_id'],c['source']['csv_sha256'],c['source']['dialogue_sha256'],c['requested_section'],c['prompt']) for c in cases]
    for name,manifest,cases,rouge in runs[1:]:
        if evidence(cases)!=evidence(first[2]): raise ValueError('Cases, order, source, section or prompt differ')
        if manifest['protocol']!=first[1]['protocol']: raise ValueError('Generation protocols differ')
        if manifest['generation_config']!=first[1]['generation_config']: raise ValueError('Generation settings differ')
        if manifest['tokenization']!=first[1]['tokenization']: raise ValueError('Input tokenization settings differ')
        if [c['model_input']['input_ids'] for c in cases]!=[c['model_input']['input_ids'] for c in first[2]]:
            raise ValueError('Actual model-visible input token IDs differ')
    per_case=[]
    for i,c in enumerate(first[2]):
        per_case.append({'case_id':c['case_id'], 'models':{
            name:{**{k:v for k,v in rouge['per_case'][i].items() if k!='case_id'},
              'prediction_sha256':__import__('hashlib').sha256(cases[i]['prediction'].encode()).hexdigest(),
              'human_factuality_status':'pending'} for name,manifest,cases,rouge in runs}})
    return {'cases':len(first[2]), 'protocol':first[1]['protocol'],
      'scope':'Paired overlap results only; no human factuality comparisons are available.',
      'models':{name:{'model':manifest['model'],'metrics':rouge['metrics'],
        'cases_sha256':manifest['cases_sha256']} for name,manifest,cases,rouge in runs},
      'per_case':per_case}
