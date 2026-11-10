"""Judge runner: resume without re-sending, per-trial jobs, freeze gate, trial merge, opaque IDs."""
import json
import tempfile
import unittest
from pathlib import Path

from factuality.claude_code_backend import CallLedger
from factuality.judge_run import freeze_judge, merge_trials, opaque_id, run_judge
from factuality.judge_v3 import rubric_hash, schema_hash

DIALOGUE = 'Patient: I am 40 years old. I have a cough.'


def case(i):
    return {'case_id': f'val:{i}', 'dialogue': DIALOGUE, 'requested_section': 'history of present illness',
            'prediction': 'The patient is 40 years old.', 'model_input': {'text': DIALOGUE, 'truncated': False}}


def judged(label='supported'):
    return {'claims_complete': True, 'section_inventory': [],
            'claims': [{'text': 'Age 40', 'fact_type': 'age', 'subject': 'patient', 'label': label,
                        'generated_quote': '40 years old', 'evidence_quotes': ['I am 40 years old'],
                        'visibility': 'visible', 'rationale': 'stated'}]}


class FakeCaller:
    def __init__(self, ledger, limit_at=None, output=None):
        self.ledger, self.limit_at, self.sent, self.output = ledger, limit_at, [], output or judged()

    def __call__(self, payload, job_id):
        self.ledger.reserve(job_id)
        self.sent.append(job_id)
        if self.limit_at is not None and len(self.sent) == self.limit_at:
            self.ledger.settle(job_id, 'usage_limited', None)
            self.ledger.block('usage_limited')
            return {'status': 'usage_limited', 'raw': None, 'usage': None, 'attempts': 1, 'cli_version': 'fake'}
        self.ledger.settle(job_id, 'complete', {'input_tokens': 1}, attempts=1)
        return {'status': 'complete', 'raw': self.output, 'usage': {'input_tokens': 1}, 'attempts': 1, 'cli_version': 'fake'}


class Runner(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ledger = CallLedger(self.tmp / 'ledger.json', cap=50)
        self.items = [('run-a', case(i)) for i in range(5)]

    def test_resume_after_usage_limit(self):
        first = FakeCaller(self.ledger, limit_at=3)
        summary = run_judge(self.items, self.tmp / 'out', split='calibration', trials=1, caller=first, ledger=self.ledger)
        self.assertEqual(summary['stopped'], 'usage_limited')
        self.assertEqual(len(first.sent), 3)
        self.ledger.unblock()
        second = FakeCaller(self.ledger)
        run_judge(self.items, self.tmp / 'out', split='calibration', trials=1, caller=second, ledger=self.ledger)
        self.assertEqual(second.sent, first.sent[2:3] + [f'calibration:{opaque_id(c["case_id"], r)}:t1' for r, c in self.items[3:]])

    def test_interrupted_reservation_recorded_and_run_continues(self):
        stuck = f'calibration:{opaque_id("val:0", "run-a")}:t1'
        self.ledger.reserve(stuck)  # simulates a crash after reserve, before settle
        caller = FakeCaller(self.ledger)
        summary = run_judge(self.items[:2], self.tmp / 'out', split='calibration', trials=1, caller=caller, ledger=self.ledger)
        self.assertIsNone(summary['stopped'])
        self.assertEqual(caller.sent, [f'calibration:{opaque_id("val:1", "run-a")}:t1'])
        record = json.loads((self.tmp / 'out' / opaque_id('val:0', 'run-a') / 't1.json').read_text())
        self.assertEqual(record['status'], 'interrupted')
        self.assertTrue(self.ledger.done(stuck))

    def test_stops_after_consecutive_content_failures(self):
        bad = judged()
        bad['claims'][0]['generated_quote'] = 'forty'
        items = [('run-a', case(i)) for i in range(8)]
        caller = FakeCaller(self.ledger, output=bad)
        summary = run_judge(items, self.tmp / 'out', split='calibration', trials=1, caller=caller, ledger=self.ledger)
        self.assertEqual(len(caller.sent), 5)
        self.assertIn('consecutive', summary['stopped'])

    def test_second_trial_only(self):
        caller = FakeCaller(self.ledger)
        run_judge(self.items[:1], self.tmp / 'out', split='calibration', trials=1, caller=caller, ledger=self.ledger)
        manifest = self.tmp / 'judge_manifest.json'
        manifest.write_text(json.dumps({'probe': {'passed': True, 'cli_version': 'fake', 'init_event_hash': 'h'}}))
        freeze_judge(manifest)
        caller2 = FakeCaller(self.ledger)
        run_judge(self.items[:1], self.tmp / 'held', split='heldout', trials=2, caller=caller2, ledger=self.ledger,
                  freeze_manifest=manifest)
        caller3 = FakeCaller(self.ledger)
        (self.tmp / 'held' / opaque_id('val:0', 'run-a') / 't2.json').unlink()
        self.ledger._locked(lambda s: s['entries'].pop(f'heldout:{opaque_id("val:0", "run-a")}:t2'))
        run_judge(self.items[:1], self.tmp / 'held', split='heldout', trials=2, caller=caller3, ledger=self.ledger,
                  freeze_manifest=manifest)
        self.assertEqual(caller3.sent, [f'heldout:{opaque_id("val:0", "run-a")}:t2'])

    def test_heldout_requires_matching_freeze(self):
        manifest = self.tmp / 'judge_manifest.json'
        manifest.write_text(json.dumps({'freeze': {'rubric_hash': 'stale', 'schema_hash': schema_hash()}}))
        with self.assertRaises(ValueError):
            run_judge(self.items, self.tmp / 'held', split='heldout', trials=2, caller=FakeCaller(self.ledger),
                      ledger=self.ledger, freeze_manifest=manifest)
        with self.assertRaises(ValueError):
            run_judge(self.items, self.tmp / 'held', split='heldout', trials=2, caller=FakeCaller(self.ledger),
                      ledger=self.ledger, freeze_manifest=None)

    def test_heldout_refuses_changed_cli_version(self):
        manifest = self.tmp / 'judge_manifest.json'
        manifest.write_text(json.dumps({'probe': {'passed': True, 'cli_version': '2.1.311', 'init_event_hash': 'h'}}))
        freeze_judge(manifest)
        with self.assertRaises(ValueError):
            run_judge(self.items, self.tmp / 'held', split='heldout', trials=2, caller=FakeCaller(self.ledger),
                      ledger=self.ledger, freeze_manifest=manifest, cli_version_now='2.2.0')

    def test_freeze_requires_passing_probe(self):
        manifest = self.tmp / 'judge_manifest.json'
        manifest.write_text(json.dumps({'probe': {'passed': False}}))
        with self.assertRaises(ValueError):
            freeze_judge(manifest)

    def test_freeze_records_hashes(self):
        manifest = self.tmp / 'judge_manifest.json'
        manifest.write_text(json.dumps({'probe': {'passed': True, 'cli_version': 'v', 'init_event_hash': 'h'}}))
        f = freeze_judge(manifest)
        self.assertEqual((f['rubric_hash'], f['schema_hash']), (rubric_hash(), schema_hash()))

    def test_invalid_output_recorded_as_failure(self):
        bad = judged()
        bad['claims'][0]['generated_quote'] = 'forty'
        run_judge(self.items[:1], self.tmp / 'out', split='calibration', trials=1,
                  caller=FakeCaller(self.ledger, output=bad), ledger=self.ledger)
        record = json.loads((self.tmp / 'out' / opaque_id('val:0', 'run-a') / 't1.json').read_text())
        self.assertEqual((record['status'], record['failure_reason']), ('content_failed', 'quote_not_found'))

    def test_only_key_has_real_ids(self):
        run_judge(self.items, self.tmp / 'out', split='calibration', trials=1, caller=FakeCaller(self.ledger), ledger=self.ledger)
        for path in (self.tmp / 'out').rglob('*.json'):
            if path.name != 'key.json':
                self.assertNotIn('val:', path.read_text(), path)
                self.assertNotIn('run-a', path.read_text(), path)
        key = json.loads((self.tmp / 'out' / 'key.json').read_text())
        self.assertEqual(key[opaque_id('val:0', 'run-a')], {'run': 'run-a', 'case_id': 'val:0'})


class Merge(unittest.TestCase):
    def result(self, label):
        from factuality.judge_v3 import build_payload, validate_judge_output
        return validate_judge_output(build_payload(case(0), 'x'), judged(label))

    def test_agreement(self):
        m = merge_trials([self.result('supported'), self.result('supported')])
        self.assertEqual((m['claims'][0]['label'], m['claims'][0]['trial_support']), ('supported', 2))

    def test_disagreement_is_unclear(self):
        m = merge_trials([self.result('supported'), self.result('contradicted')])
        self.assertEqual(m['claims'][0]['label'], 'unclear')
        self.assertTrue(m['claims'][0]['trial_disagreement'])

    def test_single_trial_claim_is_unclear(self):
        r2 = self.result('supported')
        r2['claims'] = []
        m = merge_trials([self.result('supported'), r2])
        self.assertEqual((m['claims'][0]['trial_support'], m['claims'][0]['label']), (1, 'unclear'))
        self.assertTrue(m['claims'][0]['trial_disagreement'])

    def test_different_fact_types_not_paired(self):
        r1, r2 = self.result('supported'), self.result('unsupported')
        r2['claims'][0]['fact_type'] = 'race_ethnicity'
        m = merge_trials([r1, r2])
        self.assertEqual(sorted((c['fact_type'], c['trial_support']) for c in m['claims']), [('age', 1), ('race_ethnicity', 1)])

    def test_single_trial_run_keeps_labels(self):
        m = merge_trials([self.result('contradicted')])
        self.assertEqual(m['claims'][0]['label'], 'contradicted')


class Load(unittest.TestCase):
    def test_round_trip_merges_trials_and_skips_failures(self):
        from factuality.judge_run import load_judge
        tmp = Path(tempfile.mkdtemp())
        ledger = CallLedger(tmp / 'l.json', cap=50)
        manifest = tmp / 'm.json'
        manifest.write_text(json.dumps({'probe': {'passed': True, 'cli_version': 'v', 'init_event_hash': 'h'}}))
        freeze_judge(manifest)
        items = [('run-a', case(0)), ('run-b', case(1))]
        run_judge(items, tmp / 'held', split='heldout', trials=2, caller=FakeCaller(ledger), ledger=ledger,
                  freeze_manifest=manifest)
        bad = tmp / 'held' / opaque_id('val:1', 'run-b') / 't2.json'
        record = json.loads(bad.read_text())
        record['status'] = 'content_failed'
        bad.write_text(json.dumps(record))
        merged, by_trial = load_judge(tmp / 'held')
        self.assertEqual(merged[('run-a', 'val:0')]['claims'][0]['trial_support'], 2)
        self.assertEqual(merged[('run-b', 'val:1')]['trials'], 1)
        self.assertIn(('run-a', 'val:0'), by_trial['t2'])
        self.assertNotIn(('run-b', 'val:1'), by_trial['t2'])
