"""Stable serialization and artifact identities (standard library only)."""
import hashlib
import json
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def case_hash(case):
    # Bind annotations to the entire evidence/provenance record, not just its ID.
    return digest(json.dumps(case, sort_keys=True, ensure_ascii=False))


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, 'x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def source_plan_hash(case):
    return digest(json.dumps({k:case.get(k) for k in ["case_id","source","dialogue","requested_section","reference"]},sort_keys=True,ensure_ascii=False))
