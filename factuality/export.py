"""Exact validation sampling without importing training, model, or test data."""
from .io import digest, file_hash

SECTIONS = {'CC':'chief complaint', 'GENHX':'history of present illness',
 'PASTMEDICALHX':'past medical history', 'PASTSURGICAL':'past surgeries',
 'MEDICATIONS':'medications', 'ALLERGY':'allergies', 'FAM/SOCHX':'social history',
 'EDCOURSE':'educational courses', 'ROS':'review of systems', 'EXAM':'physical exam',
 'ASSESSMENT':'assessment', 'PROCEDURES':'procedures', 'LABS':'labs', 'PLAN':'plan',
 'DISPOSITION':'disposition'}
VAL_CSV = 'MTS-Dialog/Main-Dataset/MTS-Dialog-ValidationSet.csv'


def sample_cases(csv_path=VAL_CSV, n=100, seed=42):
    import pandas as pd
    if n < 1: raise ValueError('n must be positive')
    raw = pd.read_csv(csv_path)
    rows = []
    for dialogue_id, group in raw.groupby(raw['ID'].astype(str)):
        headers = group['section_header'].tolist()
        code = str(headers[0]).upper()
        dialogue = ' '.join(group['dialogue'].tolist()).strip()
        section = SECTIONS.get(code, 'History of Present Illness')
        notebook_section = 'exam' if code == 'EXAM' else SECTIONS.get(code, '')
        reference = '\n'.join(f'{notebook_section}: {str(t).strip()}' for t in group['section_text'])
        rows.append({'schema_version': 1, 'dataset_kind': 'real', 'case_id': f'val:{dialogue_id}',
          'source': {'split':'val', 'dialogue_id':dialogue_id, 'csv':str(csv_path),
                     'csv_sha256':file_hash(csv_path), 'rows':group.index.tolist(),
                     'dialogue_sha256':digest(dialogue), 'section_headers':headers},
          'dialogue':dialogue, 'requested_section':section, 'notebook_section':notebook_section,
          'reference':reference, 'reference_sections':group['section_text'].tolist(),
          'prompt':f'Summarize the following doctor-patient dialogue into a detailed {section} clinical note: {dialogue}',
          'protocol':'flask-note', 'prediction':None, 'model_input':None, 'generation_status':'pending'})
    data = pd.DataFrame(rows).sample(n=min(n, len(rows)), random_state=seed)
    return data.to_dict('records')


def audit_cache(path, cases):
    import json
    cached = json.load(open(path))
    matches = []
    for i, pred in enumerate(cached.get('predictions', [])):
        # A reference comparison is diagnostic only; it never establishes model identity.
        ids = [c['case_id'] for c in cases if pred.get('reference') == c['reference']]
        matches.append({'prediction_index':i, 'exact_full_reference_candidate_ids':ids})
    return {'path':str(path), 'sha256':file_hash(path), 'predictions':len(matches),
      'scorable':False, 'reason':'Missing source IDs/hashes, model revision and tokenizer/generation provenance. Reference matches alone are insufficient.',
      'reference_matches':matches}
