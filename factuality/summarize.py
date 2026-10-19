"""Finalize a local run into text-free tracked metrics and a local pending queue."""
import json
from pathlib import Path
from .generate import validate_case_request
from .io import read_jsonl, file_hash, case_hash, write_json
from .review import write_queue
from .rouge import compute_rouge
from .scoring import score


def summarize_run(run_dir, out_dir, annotations_path=None, seed=42):
    root=Path(run_dir); out=Path(out_dir)
    if out.exists(): raise ValueError('Summary directory already exists; use a new path')
    manifest=json.loads((root/'generation_manifest.json').read_text())
    if manifest['status']!='complete': raise ValueError('Cannot finalize an incomplete generation run')
    if file_hash(root/'cases.jsonl')!=manifest['cases_sha256']: raise ValueError('Case artifact hash mismatch')
    cases=read_jsonl(root/'cases.jsonl')
    if len(cases)!=manifest['completed'] or [c['case_id'] for c in cases]!=manifest['case_ids']:
        raise ValueError('Manifest count/order mismatch')
    for c in cases: validate_case_request(c)
    from .export import sample_cases
    reconstructed=sample_cases(n=len(cases),seed=seed)
    for actual,expected in zip(cases,reconstructed):
        if any(actual.get(k)!=expected.get(k) for k in ['case_id','source','dialogue','prompt','reference','requested_section']):
            raise ValueError('Cases differ from freshly reconstructed validation sample/seed')
    manifest['post_run_audit']={'sampling':{'seed':seed,'n':len(cases),'method':'pandas groupby sorted string ID, sample random_state'},
      'sampling_source_prompt_verified':True,'annotation_file_sha256':file_hash(annotations_path) if annotations_path else None,
      'note':'Audit added during summarization; original generation manifest retained in local run directory.'}
    annotations=read_jsonl(annotations_path) if annotations_path else []
    metrics=score(cases,annotations)
    rouge=compute_rouge(cases)
    queue=root/'review'
    if queue.exists():
        existing=read_jsonl(queue/'annotations.jsonl')
        expected={c['case_id']:case_hash(c) for c in cases}
        if len(existing)!=len(cases) or {a['case_id']:a['case_hash'] for a in existing}!=expected:
            raise ValueError('Existing review queue differs from this run; preserve it and use a fresh run directory')
        queue_summary={'cases':len(cases),'reused_existing':True,
          'note':'Existing review work preserved; only the explicit --annotations file is scored.'}
    else:
        queue_summary=write_queue(cases,queue)
    out.mkdir(parents=True)
    write_json(out/'run_manifest.json',manifest)
    write_json(out/'metrics.json',metrics)
    write_json(out/'rouge.json',rouge)
    write_json(out/'sample_manifest.json',{'dataset_revision':manifest['dataset_revision'],
      'csv_sha256':manifest['dataset_csv_sha256'], 'sampling':manifest['post_run_audit']['sampling'], 'cases':[{
        'case_id':c['case_id'], 'source':c['source'], 'requested_section':c['requested_section'],
        'reference_sha256':__import__('hashlib').sha256(c['reference'].encode()).hexdigest(),
        'model_input_sha256':c['model_input']['sha256'], 'truncated':c['model_input']['truncated'],
        'full_token_count':c['model_input']['full_token_count']} for c in cases]})
    write_json(out/'review_queue_summary.json',queue_summary)
    return {'cases':len(cases),'human_reviewed':metrics['human']['cases_with_annotations'],
      'human_pending':metrics['human']['pending_cases'], 'truncated_cases':metrics['truncated_cases']}
