"""Blinded judge payload, output schema and strict validation (spec §6.2–6.3).

The judge sees only ALLOWED_FIELDS: no rule output, plans, construction gold,
provenance, references, run IDs or model identity.
"""
import json
from pathlib import Path

from .io import digest
from .scoring import SUBJECTS, TYPES

ALLOWED_FIELDS = ('case_id', 'dialogue', 'visible_input', 'truncated', 'requested_section', 'prediction')
LABELS = ('supported', 'contradicted', 'unsupported', 'unclear')
RUBRIC_PATH = Path(__file__).with_name('prompts') / 'judge_v3.txt'

_STR = {'type': 'string'}
OUTPUT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['claims_complete', 'claims', 'section_inventory'],
    'properties': {
        'claims_complete': {'type': 'boolean'},
        'claims': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['text', 'fact_type', 'subject', 'label', 'generated_quote', 'evidence_quotes', 'visibility', 'rationale'],
            'properties': {'text': _STR, 'fact_type': {'enum': sorted(TYPES)}, 'subject': {'enum': sorted(SUBJECTS)},
                           'label': {'enum': list(LABELS)}, 'generated_quote': _STR,
                           'evidence_quotes': {'type': 'array', 'items': _STR},
                           'visibility': {'enum': ['visible', 'truncated']}, 'rationale': _STR}}},
        'section_inventory': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['text', 'source_quote', 'status', 'generated_quote'],
            'properties': {'text': _STR, 'source_quote': _STR, 'status': {'enum': ['present', 'omitted']},
                           'generated_quote': _STR}}},
    },
}


class JudgeOutputError(ValueError):
    def __init__(self, reason, detail=''):
        super().__init__(f'{reason}: {detail}')
        self.reason = reason


def rubric_text():
    return RUBRIC_PATH.read_text(encoding='utf-8')


def rubric_hash():
    return digest(rubric_text())


def schema_hash():
    return digest(json.dumps(OUTPUT_SCHEMA, sort_keys=True))


def build_payload(case, opaque_id):
    mi = case.get('model_input') or {}
    return {'case_id': opaque_id, 'dialogue': case['dialogue'], 'visible_input': mi.get('text', ''),
            'truncated': bool(mi.get('truncated')), 'requested_section': case['requested_section'],
            'prediction': case.get('prediction') or ''}


def _occurrences(text, quote, what):
    if not isinstance(quote, str) or not quote:
        raise JudgeOutputError('quote_not_found', f'empty {what} quote')
    spans, start = [], text.find(quote)
    while start >= 0:
        spans.append({'start': start, 'end': start + len(quote), 'text': quote})
        start = text.find(quote, start + 1)
    if not spans:
        raise JudgeOutputError('quote_not_found', f'{what}: {quote[:60]!r}')
    return spans


def _require(condition, detail):
    if not condition:
        raise JudgeOutputError('schema_invalid', detail)


def validate_judge_output(payload, raw):
    _require(isinstance(raw, dict) and isinstance(raw.get('claims'), list)
             and isinstance(raw.get('section_inventory'), list) and type(raw.get('claims_complete')) is bool, 'shape')
    prediction, dialogue = payload['prediction'], payload['dialogue']
    claims, seen = [], set()
    for c in raw['claims']:
        _require(isinstance(c, dict) and c.get('fact_type') in TYPES and c.get('subject') in SUBJECTS
                 and c.get('label') in LABELS and c.get('visibility') in ('visible', 'truncated')
                 and str(c.get('text', '')).strip() and str(c.get('rationale', '')).strip()
                 and isinstance(c.get('evidence_quotes'), list), 'claim fields')
        generated = _occurrences(prediction, c.get('generated_quote'), 'summary')
        groups = [_occurrences(dialogue, q, 'source') for q in c['evidence_quotes']]
        if c['label'] in ('supported', 'contradicted') and not groups:
            raise JudgeOutputError('evidence_required', c['text'][:60])
        key = (c['fact_type'], c['subject'], c['generated_quote'], c['text'])
        if key in seen:
            raise JudgeOutputError('duplicate_claim', c['text'][:60])
        seen.add(key)
        # Only an ambiguous summary location makes the label unusable; repeated short evidence
        # (e.g. "Yes.") is flagged but keeps the judge's label.
        claims.append(dict(c, proposed_label=c['label'], label='unclear' if len(generated) > 1 else c['label'],
                           generated_span=generated[0], source_spans=[s for g in groups for s in g],
                           ambiguous_location=len(generated) > 1,
                           ambiguous_evidence=len(generated) > 1 or any(len(g) > 1 for g in groups)))
    inventory = []
    for f in raw['section_inventory']:
        _require(isinstance(f, dict) and f.get('status') in ('present', 'omitted') and str(f.get('text', '')).strip(),
                 'inventory fields')
        source = _occurrences(dialogue, f.get('source_quote'), 'inventory source')
        span = _occurrences(prediction, f.get('generated_quote'), 'inventory summary')[0] if f['status'] == 'present' else None
        inventory.append(dict(f, source_span=source[0], generated_span=span, ambiguous_evidence=len(source) > 1))
    return {'status': 'complete', 'claims_complete': raw['claims_complete'], 'claims': claims, 'section_inventory': inventory}
