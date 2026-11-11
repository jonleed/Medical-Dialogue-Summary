"""Trace actual serving outputs without changing production/training code."""
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from .io import digest, file_hash, read_jsonl, write_json


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git_revision(path='.'):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def environment():
    packages = {}
    for name in ['torch', 'transformers', 'peft', 'pandas', 'numpy', 'Flask', 'rouge-score', 'pytest']:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return {'python':platform.python_version(), 'platform':platform.platform(), 'packages':packages}


def validate_case_request(case):
    dialogue = case.get('dialogue')
    section = case.get('requested_section')
    if not isinstance(dialogue,str) or not dialogue.strip() or not isinstance(section,str) or not section.strip():
        raise ValueError('Nonempty dialogue and requested section required')
    prompt = f"Summarize the following doctor-patient dialogue into a detailed {section.strip()} clinical note: {dialogue.strip()}"
    if case.get('prompt') != prompt:
        raise ValueError('Recorded prompt differs from actual Flask request construction')
    if case.get('source',{}).get('dialogue_sha256') != digest(dialogue):
        raise ValueError('Source dialogue hash mismatch')
    return prompt


def generate(cases_path, out_dir, threads=4, base_model=None, adapter=None):
    import torch
    import pandas as pd
    from unittest.mock import patch
    out = Path(out_dir)
    manifest_path = out / 'generation_manifest.json'
    if manifest_path.exists(): raise ValueError('Use a new directory for each generation run; no overwrite/resume')
    out.mkdir(parents=True, exist_ok=True)
    if threads < 1: raise ValueError('threads must be positive')
    if base_model: os.environ['BASE_MODEL_ID'] = base_model
    if adapter and adapter.lower() != 'none': os.environ['ADAPTER_MODEL_ID'] = adapter
    cases = read_jsonl(cases_path)
    if not cases: raise ValueError('Empty generation case set')
    for case in cases: validate_case_request(case)
    if len({c['case_id'] for c in cases}) != len(cases): raise ValueError('Duplicate case IDs')
    if any(c['protocol'] != 'flask-note' or c['source']['split'] != 'val' for c in cases):
        raise ValueError('Only exploratory validation Flask protocol is supported')
    manifest = {'schema_version':1, 'started_at':utc_now(), 'status':'running',
      'protocol':'flask-note', 'code_revision':git_revision(), 'dataset_revision':git_revision('MTS-Dialog'),
      'environment':environment(), 'command':__import__('sys').argv,
      'export_sha256':file_hash(cases_path), 'case_ids':[c['case_id'] for c in cases],
      'human_pilot_count':0, 'human_reviewed_count':0, 'human_rubric_frozen':False,
      'threads':threads, 'completed':0, 'source_code_hashes':{
          str(p):file_hash(p) for p in [Path('app.py'), *sorted(Path('factuality').glob('*.py'))]},
      'dataset_csv_sha256':cases[0]['source']['csv_sha256'] if cases else None,
      'test_data_import':'Suppressed dataset-table sample initialization only; no test CSV is read.'}
    write_json(manifest_path, manifest)
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    try:
        # app.SAMPLES is a UI-only test-set table, irrelevant to the generation route.
        # Avoid reading test data on module import, without altering production files.
        original_read_csv = pd.read_csv
        def validation_only_read(path, *args, **kwargs):
            if str(path).endswith('MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv'):
                return pd.DataFrame(columns=['ID','dialogue','section_header','section_text'])
            return original_read_csv(path, *args, **kwargs)
        with patch.object(pd, 'read_csv', validation_only_read):
            import app as serving
        if serving.model is not None:
            raise ValueError('Run generation in a fresh process to avoid reusing a previously loaded model')
        if adapter and adapter.lower() == 'none':
            # Same Flask route and decode; substitute only the baseline weight loader.
            def load_base_only():
                if serving.model is None:
                    rev = getattr(serving, 'BASE_MODEL_REVISION', None)
                    serving.tokenizer = serving.AutoTokenizer.from_pretrained(serving.BASE_MODEL_ID, token=serving.HF_TOKEN, revision=rev)
                    serving.model = serving.AutoModelForSeq2SeqLM.from_pretrained(serving.BASE_MODEL_ID, token=serving.HF_TOKEN, revision=rev).eval()
            serving.load_model = load_base_only
        serving.load_model()
        tokenizer = serving.tokenizer
        base_path = Path(serving.BASE_MODEL_ID)
        base_revision = getattr(serving.model.config, '_commit_hash', None)
        if base_revision is None and base_path.parent.name == 'snapshots':
            base_revision = base_path.name
        local_base_hashes = {p.name:file_hash(p) for p in base_path.iterdir() if p.is_file()} if base_path.is_dir() else {}
        adapter_path = None if adapter and adapter.lower() == 'none' else Path(serving.ADAPTER_MODEL_ID)
        if adapter_path is not None and not adapter_path.is_dir():
            raise ValueError('For reproducibility this harness requires a local adapter directory')
        manifest.update({'model':{'base_model':serving.BASE_MODEL_ID, 'resolved_base_revision':base_revision,
           'local_base_file_hashes':local_base_hashes,
           'adapter':str(adapter_path) if adapter_path else None, 'adapter_hashes':{
               p.name:file_hash(p) for p in sorted(adapter_path.iterdir()) if p.is_file()} if adapter_path else {},
           'loader':'base-only substitution' if adapter_path is None else 'app LoRA merge',
           'device':str(serving.model.device)},
          'tokenizer':{'name':tokenizer.name_or_path, 'class':type(tokenizer).__name__,
            'backend_sha256':digest(tokenizer.backend_tokenizer.to_str()),
            'special_tokens':{k:str(v) for k,v in tokenizer.special_tokens_map.items()},
            'padding_side':tokenizer.padding_side, 'truncation_side':tokenizer.truncation_side},
          'generation_config':serving.generation_config.to_dict(),
          'model_default_generation_config':serving.model.generation_config.to_dict(),
          'tokenization':{'max_length':serving.INPUT_MAX_LENGTH, 'padding':'max_length', 'truncation':True}})
        write_json(manifest_path, manifest)
        client = serving.app.test_client()
        with open(out / 'cases.jsonl', 'x', encoding='utf-8') as f:
            for i, c in enumerate(cases):
                start = time.monotonic()
                response = client.post('/generate_note', json={'dialogue':c['dialogue'], 'primary':c['requested_section']})
                if response.status_code != 200:
                    raise RuntimeError(f"{c['case_id']}: HTTP {response.status_code}: {response.get_json()}")
                payload = response.get_json()
                if payload['tokens'] != len(tokenizer(c['prompt'])['input_ids']):
                    raise ValueError('Served tokenizer count differs from reconstructed prompt')
                tokens = tokenizer(c['prompt'], max_length=serving.INPUT_MAX_LENGTH,
                    padding='max_length', truncation=True)
                c['model_input'] = {'input_ids':tokens['input_ids'], 'attention_mask':tokens['attention_mask'],
                    'text':tokenizer.decode([t for t,m in zip(tokens['input_ids'], tokens['attention_mask']) if m], skip_special_tokens=True),
                    'full_token_count':payload['tokens'], 'truncated':payload['truncated'],
                    'sha256':digest(json.dumps(tokens['input_ids']))}
                c.update(prediction=payload['note'], generation_status='generated',
                         provenance={'manifest':'generation_manifest.json', 'base_revision':base_revision,
                          'adapter_weights_sha256':manifest['model']['adapter_hashes'].get('adapter_model.safetensors'),
                          'generated_at':utc_now(), 'seconds':round(time.monotonic()-start,3)})
                f.write(json.dumps(c, ensure_ascii=False)+'\n'); f.flush()
                manifest['completed'] = i+1
                write_json(manifest_path, manifest)
                print(f"{i+1}/{len(cases)} {c['case_id']} {c['provenance']['seconds']}s", flush=True)
        manifest['status'] = 'complete'
    except KeyboardInterrupt:
        manifest['status'] = 'interrupted'
        manifest['error'] = 'KeyboardInterrupt; completed cases are preserved, remaining cases were not generated.'
        raise
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest['finished_at'] = utc_now()
        if (out/'cases.jsonl').exists(): manifest['cases_sha256'] = file_hash(out/'cases.jsonl')
        write_json(manifest_path, manifest)
