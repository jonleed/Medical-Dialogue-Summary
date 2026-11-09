"""Blind source-only plans, plan freeze, model-blinded labelling packet, and label import (spec §5.2).

The labelling page never receives rule output, judge output, run IDs or model
names; the item → (run, case) key lives in a separate key.json the page never loads.
"""
import json
import random
import time
from pathlib import Path

from .io import case_hash, file_hash, read_jsonl, source_plan_hash, write_jsonl
from .review import blank_annotation, blind_plans
from .scoring import SUBJECTS, TYPES, validate_annotation, validate_span

TEMPLATE = Path(__file__).with_name('templates_v3') / 'label.html'
FREEZE_FIELDS = ('expected_facts_frozen', 'expected_facts_blind')


def _render(path, mode, items):
    payload = {'mode': mode, 'items': items}
    payload['packet_hash'] = case_hash(payload)[:16]
    data = json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')
    Path(path).write_text(TEMPLATE.read_text(encoding='utf-8').replace('/*__DATA__*/', data), encoding='utf-8')


def _new_dir(out_dir):
    out = Path(out_dir)
    if out.exists():
        raise ValueError(f'{out} exists; use a new directory')
    out.mkdir(parents=True)
    return out


def write_plan_packet(cases, out_dir):
    out = _new_dir(out_dir)
    plans = blind_plans(cases)
    write_jsonl(out / 'plans.jsonl', plans)
    _render(out / 'plans.html', 'plan', [{'case_id': p['case_id'], 'requested_section': p['requested_section'],
                                          'dialogue': p['dialogue'], 'reference': p['reference'], 'plan': p}
                                         for p in plans])
    return out


def _check_plan(plan):
    for field in FREEZE_FIELDS:
        if plan.get(field) is not True:
            raise ValueError(f"{plan['case_id']}: {field} must be true")
    if plan.get('expected_facts_reviewer_kind') != 'human' or not str(plan.get('expected_facts_reviewer', '')).strip():
        raise ValueError(f"{plan['case_id']}: human reviewer identity required")
    if not plan.get('expected_facts_frozen_at'):
        raise ValueError(f"{plan['case_id']}: freeze time required")
    ids = set()
    for fact in plan.get('expected_facts', []):
        if not fact.get('id') or fact['id'] in ids:
            raise ValueError(f"{plan['case_id']}: duplicate or missing fact id")
        ids.add(fact['id'])
        if fact.get('fact_type') not in TYPES or fact.get('subject', 'patient') not in SUBJECTS:
            raise ValueError(f"{plan['case_id']}/{fact['id']}: invalid type or subject")
        if fact.get('label') not in {'pending', 'excluded'} or not str(fact.get('text', '')).strip():
            raise ValueError(f"{plan['case_id']}/{fact['id']}: plan facts are pending or excluded with text")
        if fact['label'] == 'pending' and not fact.get('source_spans'):
            raise ValueError(f"{plan['case_id']}/{fact['id']}: expected fact needs a source span")
        for span in fact.get('source_spans', []):
            validate_span(plan['dialogue'], span)


def freeze_plans(plans_path):
    plans_path = Path(plans_path)
    plans = read_jsonl(plans_path)
    for plan in plans:
        _check_plan(plan)
    record = {'hash': file_hash(plans_path), 'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'n': len(plans)}
    (plans_path.parent / 'plans_freeze.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def write_label_packet(cases_by_model, plans_path, freeze_path, out_dir, seed=20261109):
    freeze = json.loads(Path(freeze_path).read_text())
    if freeze['hash'] != file_hash(plans_path):
        raise ValueError('plans changed after freeze; re-freeze is not allowed once summaries may have been seen')
    plans = {p['case_id']: p for p in read_jsonl(plans_path)}
    entries = []
    for run, cases in cases_by_model.items():
        for case in cases:
            plan = plans.get(case['case_id'])
            if plan is None:
                continue
            if plan['source_plan_hash'] != source_plan_hash(case):
                raise ValueError(f"stale plan for {case['case_id']}")
            entries.append((run, case, plan))
    random.Random(seed).shuffle(entries)
    out = _new_dir(out_dir)
    items, key = [], {}
    for n, (run, case, plan) in enumerate(entries, 1):
        item_id = f'item-{n:03d}'
        mi = case.get('model_input') or {}
        items.append({'item_id': item_id, 'requested_section': case['requested_section'], 'dialogue': case['dialogue'],
                      'visible_input': mi.get('text', ''), 'truncated': bool(mi.get('truncated')),
                      'prediction': case.get('prediction') or '',
                      'plan_facts': [{k: f[k] for k in ('id', 'text', 'fact_type', 'label', 'source_spans')}
                                     for f in plan['expected_facts']]})
        key[item_id] = {'run': run, 'case_id': case['case_id'], 'plan': plan}
    (out / 'items.json').write_text(json.dumps(items, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    (out / 'key.json').write_text(json.dumps(key, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    _render(out / 'label.html', 'label', items)
    return out


def import_labels(labels_jsonl, key_path, cases_by_model):
    key = json.loads(Path(key_path).read_text(encoding='utf-8'))
    records = []
    for row in read_jsonl(labels_jsonl):
        k = key[row['item_id']]
        case = next(c for c in cases_by_model[k['run']] if c['case_id'] == k['case_id'])
        plan = k['plan']
        labels = {f['id']: f for f in row.get('expected_facts', [])}
        facts = []
        for f in plan['expected_facts']:
            got = labels.get(f['id'], {})
            merged = dict(f, label='excluded' if f['label'] == 'excluded' else got.get('label', 'pending'),
                          rationale=got.get('rationale') or f.get('rationale', ''))
            if got.get('generated_span'):
                merged['generated_span'] = got['generated_span']
            facts.append(merged)
        record = blank_annotation(case)
        record.update(run=k['run'], reviewer=row['reviewer'], reviewer_kind='human', status=row['status'],
                      claims_complete=bool(row['claims_complete']), claims=row['claims'], expected_facts=facts,
                      unresolved=row.get('unresolved', []),
                      **{f: plan[f] for f in ('expected_facts_complete', 'expected_facts_frozen', 'expected_facts_blind',
                                              'expected_facts_reviewer_kind', 'expected_facts_reviewer',
                                              'expected_facts_frozen_at')})
        validate_annotation(case, record)
        records.append(record)
    return records
