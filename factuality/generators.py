"""Produce benchmark summaries with any configured summarizer.

Output is JSONL, one record per case: {"case_id", "prediction", "model_input": {"text", "truncated",
"full_token_count"}}. The `model_input` block lets the rule checker separate errors the model could
have avoided from evidence that was cut off by its input window.

Generators:
- flask:   this repository's serving route (factuality.generate), same pipeline as the web app.
- hf:      any Hugging Face seq2seq model at a pinned revision, with the repository's decoding settings.
- command: an external program; cases go to its stdin as JSONL {"case_id", "dialogue",
           "requested_section", "prompt"}, and it must print JSONL {"case_id", "prediction"}.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .io import digest, read_jsonl

INPUT_MAX_LENGTH = 512
DECODING = {'do_sample': False, 'num_beams': 4, 'repetition_penalty': 2.0, 'no_repeat_ngram_size': 4, 'max_length': 512}


def prompt_for(case):
    return (f"Summarize the following doctor-patient dialogue into a detailed {case['requested_section'].strip()} "
            f"clinical note: {case['dialogue'].strip()}")


def _check_ids(cases, rows):
    want = [c['case_id'] for c in cases]
    got = [r.get('case_id') for r in rows]
    if len(set(got)) != len(got):
        raise ValueError('generator returned duplicate case_ids')
    missing = [i for i in want if i not in set(got)]
    if missing:
        raise ValueError(f'generator returned no summary for {missing[:5]}' + (' …' if len(missing) > 5 else ''))
    by_id = {r['case_id']: r for r in rows}
    return [by_id[i] for i in want]


def _command(spec, cases):
    stdin = ''.join(json.dumps({'case_id': c['case_id'], 'dialogue': c['dialogue'],
                                'requested_section': c['requested_section'], 'prompt': prompt_for(c)}) + '\n' for c in cases)
    proc = subprocess.run(spec.command, input=stdin, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(f'{spec.name}: command exited {proc.returncode}: {proc.stderr.strip()[:300]}')
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    return [{'case_id': r['case_id'], 'prediction': r['prediction']} for r in _check_ids(cases, rows)]


def _hf(spec, cases):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec.base_model, revision=spec.base_revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(spec.base_model, revision=spec.base_revision)
    if spec.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, spec.adapter, revision=spec.adapter_revision).merge_and_unload()
    model.eval()
    rows = []
    for case in cases:
        prompt = prompt_for(case)
        full = len(tokenizer(prompt)['input_ids'])
        inputs = tokenizer(prompt, max_length=INPUT_MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt')
        out = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], **DECODING)
        visible = [i for i, m in zip(inputs['input_ids'][0], inputs['attention_mask'][0]) if int(m)]
        rows.append({'case_id': case['case_id'], 'prediction': tokenizer.decode(out[0], skip_special_tokens=True),
                     'model_input': {'text': tokenizer.decode(visible, skip_special_tokens=True),
                                     'truncated': full > INPUT_MAX_LENGTH, 'full_token_count': full}})
    return rows


def _flask(spec, cases):
    from .generate import generate
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        requests = [dict(c, prompt=prompt_for(c), protocol='flask-note',
                         source=dict(c.get('source') or {'split': 'val'}, dialogue_sha256=digest(c['dialogue'])))
                    for c in cases]
        (tmp / 'cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in requests))
        os.environ['BASE_MODEL_REVISION'] = spec.base_revision
        if spec.adapter_revision:
            os.environ['ADAPTER_REVISION'] = spec.adapter_revision
        adapter = spec.adapter or 'none'
        if spec.adapter and not Path(spec.adapter).is_dir():
            # The Flask harness needs a local adapter folder: resolve the hub ID to its pinned snapshot.
            from huggingface_hub import snapshot_download
            try:
                adapter = snapshot_download(spec.adapter, revision=spec.adapter_revision, local_files_only=True)
            except Exception:  # not cached yet
                adapter = snapshot_download(spec.adapter, revision=spec.adapter_revision)
        generate(tmp / 'cases.jsonl', tmp / 'run', base_model=spec.base_model, adapter=adapter)
        rows = read_jsonl(tmp / 'run' / 'cases.jsonl')
    return [{'case_id': r['case_id'], 'prediction': r['prediction'],
             'model_input': {k: r['model_input'][k] for k in ('text', 'truncated', 'full_token_count')}}
            for r in _check_ids(cases, rows)]


def generate_summaries(spec, cases, out_path):
    rows = {'command': _command, 'hf': _hf, 'flask': _flask}[spec.generator](spec, cases)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    return len(rows)
