"""Blind plan freeze, model-blinded labelling packet, and label import back to canonical annotations."""
import json
import tempfile
import unittest
from pathlib import Path

from factuality.annotate import freeze_plans, import_labels, write_label_packet, write_plan_packet
from factuality.io import read_jsonl

D1 = 'Doctor: How old are you? Patient: I am 40 years old. I have a cough.'
D2 = 'Doctor: Any allergies? Patient: I am allergic to penicillin.'
FORBIDDEN = ['BioBART', 'BART', 'biobart', 'lora', 'proposed_label', 'rule_', 'suggestion', 'run-biobart', 'run-bart']


def case(cid, dialogue, prediction):
    return {'case_id': cid, 'dialogue': dialogue, 'requested_section': 'history of present illness',
            'reference': 'ref', 'source': {'split': 'val'}, 'prediction': prediction,
            'model_input': {'text': dialogue, 'truncated': False}}


CASES = {
    'run-biobart': [case('val:1', D1, 'The patient is 40 years old with cough.'), case('val:2', D2, 'Penicillin allergy.')],
    'run-bart': [case('val:1', D1, 'The patient is 41 years old.'), case('val:2', D2, 'No known allergies.')],
}


def frozen_plan(plan, dialogue):
    start = dialogue.index('cough') if 'cough' in dialogue else dialogue.index('penicillin')
    phrase = dialogue[start:start + 5]
    plan.update(expected_facts=[{'id': 'f1', 'text': 'Symptom', 'fact_type': 'symptom', 'subject': 'patient',
                                 'label': 'pending', 'visibility': 'visible', 'rationale': 'in section',
                                 'source_spans': [{'start': start, 'end': start + 5, 'text': phrase}],
                                 'visible_spans': [{'start': start, 'end': start + 5, 'text': phrase}]}],
                expected_facts_complete=True, expected_facts_frozen=True, expected_facts_blind=True,
                expected_facts_reviewer_kind='human', expected_facts_reviewer='Owner',
                expected_facts_frozen_at='2026-11-10T00:00:00Z')
    return plan


class PlanFreeze(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        write_plan_packet(CASES['run-biobart'], self.tmp / 'plans')
        self.plans_path = self.tmp / 'plans' / 'plans.jsonl'

    def write(self, plans):
        self.plans_path.write_text(''.join(json.dumps(p) + '\n' for p in plans))

    def frozen(self):
        plans = read_jsonl(self.plans_path)
        by_id = {c['case_id']: c['dialogue'] for c in CASES['run-biobart']}
        self.write([frozen_plan(p, by_id[p['case_id']]) for p in plans])
        return freeze_plans(self.plans_path)

    def test_plan_page_is_source_only(self):
        CASES['run-biobart'][0]['reference'] = 'REFERENCE NOTE TEXT'
        try:
            write_plan_packet(CASES['run-biobart'], self.tmp / 'plans2')
        finally:
            CASES['run-biobart'][0]['reference'] = 'ref'
        self.assertNotIn('REFERENCE NOTE TEXT', (self.tmp / 'plans2' / 'plans.html').read_text())

    def test_plan_packet_has_no_predictions(self):
        text = (self.tmp / 'plans' / 'plans.html').read_text() + self.plans_path.read_text()
        self.assertNotIn('40 years old with cough', text)
        self.assertNotIn('Penicillin allergy', text)

    def test_unfrozen_plan_rejected(self):
        with self.assertRaises(ValueError):
            freeze_plans(self.plans_path)

    def test_bad_plan_span_rejected(self):
        plans = read_jsonl(self.plans_path)
        p = frozen_plan(plans[0], D1 if plans[0]['case_id'] == 'val:1' else D2)
        p['expected_facts'][0]['source_spans'][0]['text'] = 'xxxxx'
        self.write([p] + plans[1:])
        with self.assertRaises(ValueError):
            freeze_plans(self.plans_path)

    def test_packet_refuses_changed_plans(self):
        self.frozen()
        plans = read_jsonl(self.plans_path)
        plans[0]['expected_facts'][0]['text'] = 'edited after freeze'
        self.write(plans)
        with self.assertRaises(ValueError):
            write_label_packet(CASES, self.plans_path, self.tmp / 'plans' / 'plans_freeze.json', self.tmp / 'label')

    def test_packet_is_blind_and_shuffled(self):
        self.frozen()
        out = self.tmp / 'label'
        write_label_packet(CASES, self.plans_path, self.tmp / 'plans' / 'plans_freeze.json', out)
        page = (out / 'label.html').read_text() + (out / 'items.json').read_text()
        for word in FORBIDDEN:
            self.assertNotIn(word, page, word)
        items = json.loads((out / 'items.json').read_text())
        key = json.loads((out / 'key.json').read_text())
        order = [(key[i['item_id']]['run'], key[i['item_id']]['case_id']) for i in items]
        natural = [(r, c['case_id']) for r, cs in CASES.items() for c in cs]
        self.assertNotEqual(order, natural)
        self.assertEqual(sorted(order), sorted(natural))
        write_label_packet(CASES, self.plans_path, self.tmp / 'plans' / 'plans_freeze.json', self.tmp / 'label2')
        self.assertEqual((self.tmp / 'label2' / 'items.json').read_text(), (out / 'items.json').read_text())

    def test_import_labels(self):
        self.frozen()
        out = self.tmp / 'label'
        write_label_packet(CASES, self.plans_path, self.tmp / 'plans' / 'plans_freeze.json', out)
        key = json.loads((out / 'key.json').read_text())
        item = next(i for i, k in key.items() if k['run'] == 'run-bart' and k['case_id'] == 'val:1')
        pred = 'The patient is 41 years old.'
        label = {'item_id': item, 'reviewer': 'Owner', 'status': 'partial', 'claims_complete': True, 'unresolved': [],
                 'claims': [{'id': 'c1', 'text': 'Patient is 41', 'fact_type': 'age', 'subject': 'patient',
                             'label': 'contradicted', 'visibility': 'visible', 'rationale': 'Source says 40.',
                             'generated_span': {'start': 15, 'end': 17, 'text': '41'},
                             'source_spans': [{'start': D1.index('40'), 'end': D1.index('40') + 2, 'text': '40'}],
                             'visible_spans': [{'start': D1.index('40'), 'end': D1.index('40') + 2, 'text': '40'}]}],
                 'expected_facts': [{'id': 'f1', 'label': 'omitted', 'rationale': 'cough absent'}]}
        labels = self.tmp / 'labels.jsonl'
        labels.write_text(json.dumps(label) + '\n')
        [record] = import_labels(labels, out / 'key.json', CASES)
        self.assertEqual((record['case_id'], record['run'], record['reviewer_kind']), ('val:1', 'run-bart', 'human'))
        self.assertEqual(record['expected_facts'][0]['label'], 'omitted')
        self.assertEqual(record['expected_facts'][0]['fact_type'], 'symptom')
        label['claims'][0]['generated_span']['text'] = '40'
        labels.write_text(json.dumps(label) + '\n')
        with self.assertRaises(ValueError):
            import_labels(labels, out / 'key.json', CASES)
        self.assertEqual(pred[15:17], '41')


if __name__ == '__main__':
    unittest.main()


class VisibleSpansRecomputed(unittest.TestCase):
    def test_plan_fact_visible_span_mapped_into_model_input(self):
        prefix = 'Summarize the following doctor-patient dialogue into a detailed history of present illness clinical note: '
        cases = {r: [dict(c, model_input={'text': prefix + c['dialogue'], 'truncated': False}) for c in cs]
                 for r, cs in CASES.items()}
        cases['run-bart'][0]['model_input'] = {'text': prefix + D1[:20], 'truncated': True}  # cough beyond window
        tmp = Path(tempfile.mkdtemp())
        write_plan_packet(cases['run-biobart'], tmp / 'plans')
        plans_path = tmp / 'plans' / 'plans.jsonl'
        by_id = {c['case_id']: c['dialogue'] for c in cases['run-biobart']}
        plans_path.write_text(''.join(json.dumps(frozen_plan(p, by_id[p['case_id']])) + '\n' for p in read_jsonl(plans_path)))
        freeze_plans(plans_path)
        write_label_packet(cases, plans_path, tmp / 'plans' / 'plans_freeze.json', tmp / 'label')
        key = json.loads((tmp / 'label' / 'key.json').read_text())
        rows = [{'item_id': i, 'reviewer': 'Owner', 'status': 'partial', 'claims_complete': False, 'unresolved': ['x'],
                 'claims': [], 'expected_facts': [{'id': 'f1', 'label': 'omitted', 'rationale': 'absent'}]} for i in key]
        labels = tmp / 'labels.jsonl'
        labels.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        records = {(r['run'], r['case_id']): r for r in import_labels(labels, tmp / 'label' / 'key.json', cases)}
        visible = records[('run-biobart', 'val:1')]['expected_facts'][0]
        self.assertEqual(visible['visibility'], 'visible')
        self.assertEqual(visible['visible_spans'][0]['start'], len(prefix) + D1.index('cough'))
        cut = records[('run-bart', 'val:1')]['expected_facts'][0]
        self.assertEqual((cut['visibility'], cut['visible_spans']), ('truncated', []))
