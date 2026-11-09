"""Judge payload allowlist/blinding and strict validation of judge output."""
import json
import unittest

from factuality.judge_v3 import (ALLOWED_FIELDS, OUTPUT_SCHEMA, JudgeOutputError, build_payload, rubric_hash,
                                 schema_hash, validate_judge_output)

FULL = {'case_id': 'val:7', 'dataset_kind': 'counterfactual', 'dialogue': 'Patient: I am 40 years old. I have a cough.',
        'prediction': 'The patient is 40 years old.', 'requested_section': 'history of present illness',
        'model_input': {'text': 'Patient: I am 40 years old.', 'truncated': True, 'input_ids': [1, 2]},
        'reference': 'gold note', 'source': {'split': 'val'}, 'provenance': {'model': 'BioBART'},
        'rule_proposals': [{'proposed_label': 'contradicted'}], 'suggestions': [], 'plans': [], 'expected_facts': [],
        'counterfactual': {'gold': {'age': 40}}, 'model': 'BioBART', 'run': 'biobart-flask'}


def claim(**kw):
    c = {'text': 'Patient is 40', 'fact_type': 'age', 'subject': 'patient', 'label': 'supported',
         'generated_quote': '40 years old', 'evidence_quotes': ['I am 40 years old'], 'rationale': 'stated',
         'visibility': 'visible'}
    c.update(kw)
    return c


def raw(claims=None, inventory=None):
    return {'claims_complete': True, 'claims': claims if claims is not None else [claim()],
            'section_inventory': inventory if inventory is not None else []}


class Payload(unittest.TestCase):
    def test_exact_allowlist(self):
        p = build_payload(FULL, 'x1')
        self.assertEqual(set(p), set(ALLOWED_FIELDS))
        self.assertEqual(p['case_id'], 'x1')
        self.assertEqual((p['visible_input'], p['truncated']), ('Patient: I am 40 years old.', True))

    def test_blinding(self):
        text = json.dumps(build_payload(FULL, 'x1'))
        for leak in ['rule_proposals', 'suggestions', 'plans', 'expected_facts', 'counterfactual', 'provenance',
                     'reference', 'BioBART', 'biobart', 'val:7', 'gold', 'input_ids']:
            self.assertNotIn(leak, text, leak)

    def test_hashes_and_schema(self):
        self.assertEqual(len(rubric_hash()), 64)
        self.assertEqual(len(schema_hash()), 64)
        self.assertIn('race_ethnicity', json.dumps(OUTPUT_SCHEMA))


class Validate(unittest.TestCase):
    def setUp(self):
        self.p = build_payload(FULL, 'x1')

    def reason(self, payload, data):
        with self.assertRaises(JudgeOutputError) as e:
            validate_judge_output(payload, data)
        return e.exception.reason

    def test_valid(self):
        out = validate_judge_output(self.p, raw())
        self.assertEqual(out['status'], 'complete')
        self.assertEqual(out['claims'][0]['generated_span']['text'], '40 years old')
        self.assertEqual(out['claims'][0]['source_spans'][0]['text'], 'I am 40 years old')

    def test_whitespace_mismatch_is_quote_not_found(self):
        p = dict(self.p, prediction='The patient  is 40.')
        self.assertEqual(self.reason(p, raw([claim(generated_quote='The patient is 40.')])), 'quote_not_found')

    def test_missing_source_quote(self):
        self.assertEqual(self.reason(self.p, raw([claim(evidence_quotes=['I am 41'])])), 'quote_not_found')

    def test_supported_requires_evidence(self):
        self.assertEqual(self.reason(self.p, raw([claim(evidence_quotes=[])])), 'evidence_required')

    def test_unsupported_without_evidence_ok(self):
        validate_judge_output(self.p, raw([claim(label='unsupported', evidence_quotes=[])]))

    def test_schema_invalid(self):
        self.assertEqual(self.reason(self.p, {'claims': 'x'}), 'schema_invalid')
        self.assertEqual(self.reason(self.p, raw([claim(fact_type='race')])), 'schema_invalid')

    def test_duplicate(self):
        self.assertEqual(self.reason(self.p, raw([claim(), claim()])), 'duplicate_claim')

    def test_repeated_quote_is_ambiguous_unclear(self):
        p = dict(self.p, prediction='40 years old. Still 40 years old.')
        out = validate_judge_output(p, raw())
        self.assertEqual((out['claims'][0]['label'], out['claims'][0]['ambiguous_evidence']), ('unclear', True))

    def test_repeated_evidence_keeps_label(self):
        p = dict(self.p, dialogue='Patient: I am 40 years old. Doctor: So I am 40 years old, you said?')
        out = validate_judge_output(p, raw([claim(evidence_quotes=['I am 40 years old'])]))
        self.assertEqual((out['claims'][0]['label'], out['claims'][0]['ambiguous_evidence']), ('supported', True))

    def test_inventory(self):
        inv = [{'text': 'Cough', 'source_quote': 'I have a cough', 'status': 'omitted', 'generated_quote': ''},
               {'text': 'Age', 'source_quote': 'I am 40 years old', 'status': 'present', 'generated_quote': '40 years old'}]
        out = validate_judge_output(self.p, raw(inventory=inv))
        self.assertIsNone(out['section_inventory'][0]['generated_span'])
        self.assertEqual(out['section_inventory'][1]['generated_span']['text'], '40 years old')
        bad = [dict(inv[1], generated_quote='forty')]
        self.assertEqual(self.reason(self.p, raw(inventory=bad)), 'quote_not_found')


if __name__ == '__main__':
    unittest.main()
