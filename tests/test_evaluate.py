"""Evaluate any summarizer against the published benchmark (fake judge; no live calls)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factuality.claude_code_backend import CallLedger
from factuality.evaluate import evaluate_model
from factuality.judge import rubric_hash, schema_hash
from factuality.rebuild import Bundle
from test_judge_run import FakeCaller, judged

D = 'Patient: I am 40 years old. I have a cough.'
PRED = 'The patient is 40 years old.'


def span(text, phrase):
    s = text.index(phrase)
    return {'start': s, 'end': s + len(phrase), 'text': phrase}


def bundle():
    cases = {f'val:{i}': {'case_id': f'val:{i}', 'dialogue': D, 'requested_section': 'history of present illness',
                          'source': {'split': 'val'}} for i in (1, 2, 3)}
    plans = [{'case_id': i, 'expected_facts': [{'id': 'f1', 'label': 'pending', 'source_spans': [span(D, 'cough')]},
                                               {'id': 'f2', 'label': 'excluded', 'source_spans': [span(D, '40 years old')]}]}
             for i in cases]
    return Bundle(cases=cases, summaries={}, cf_cases=[], cf_summaries={}, plans=plans, labels=[], splits={},
                  judge_config={'freeze': {'rubric_hash': rubric_hash(), 'schema_hash': schema_hash()}})


RELIABILITY = {'unsupported_claim': {'precision': 0.82, 'recall': 0.56}, 'omission': {'precision': 0.93, 'recall': 0.87}}


class Evaluate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.summaries = [{'case_id': 'val:1', 'prediction': PRED, 'model_input': {'text': D, 'truncated': False}},
                          {'case_id': 'val:2', 'prediction': PRED, 'model_input': {'text': D, 'truncated': False}},
                          {'case_id': 'val:99', 'prediction': PRED}]

    def run_eval(self, caller=None, **kw):
        ledger = CallLedger(self.tmp / 'ledger.json', cap=20)
        return evaluate_model('m', self.summaries, bundle(), caller=caller(ledger) if caller else None, ledger=ledger,
                              work_dir=self.tmp, reliability=RELIABILITY, **kw)

    def test_coverage(self):
        r = self.run_eval()
        self.assertEqual(r['coverage'], {'scored': 2, 'missing': 1, 'unknown': 1})
        self.assertEqual(r['validity']['summary'], 2)

    def test_without_model_input_visibility_unclear(self):
        self.summaries[0].pop('model_input')
        self.summaries[0]['prediction'] = 'This is a 50-year-old white male.'
        r = self.run_eval()
        self.assertGreater(sum(r['rules']['by_type'].get('age', {}).values()), 0)
        self.assertEqual(r['rules']['visibility'].get('truncated', 0), 0)
        self.assertGreater(r['rules']['visibility'].get('unclear', 0), 0)

    def test_no_judge(self):
        self.assertIsNone(self.run_eval()['judge'])

    def test_judge_rates_and_reliability(self):
        out = judged()
        out['section_inventory'] = [{'text': 'cough', 'source_quote': 'I have a cough', 'status': 'omitted', 'generated_quote': ''}]
        r = self.run_eval(caller=lambda ledger: FakeCaller(ledger, output=out))
        j = r['judge']
        self.assertEqual(j['rates']['omitted'], {'numerator': 2, 'denominator': 2, 'rate': 1.0, 'unassessed': 0})
        self.assertEqual(j['rates']['contradicted']['rate'], 0.0)
        self.assertEqual(j['reliability'], RELIABILITY)
        self.assertEqual(j['judged'], 2)

    def test_rubric_changed_since_freeze(self):
        with patch('factuality.evaluate.rubric_hash', return_value='edited'):
            with self.assertRaises(ValueError) as e:
                self.run_eval(caller=lambda ledger: FakeCaller(ledger))
        self.assertIn('changed since the published freeze', str(e.exception))
        self.assertIn('rubric_hash', str(e.exception))

    def test_cli_version_drift_refused_unless_accepted(self):
        b = bundle()
        b.judge_config['freeze']['cli_version'] = '2.1.311 (Claude Code)'
        ledger = CallLedger(self.tmp / 'l2.json', cap=20)
        with self.assertRaises(ValueError) as e:
            evaluate_model('m', self.summaries, b, caller=FakeCaller(ledger), ledger=ledger, work_dir=self.tmp,
                           reliability=RELIABILITY, cli_version_now='3.0.0 (Claude Code)')
        self.assertIn('cli_version', str(e.exception))
        r = evaluate_model('m', self.summaries, b, caller=FakeCaller(ledger), ledger=ledger, work_dir=self.tmp,
                           reliability=RELIABILITY, cli_version_now='3.0.0 (Claude Code)', accept_judge_drift=True)
        self.assertFalse(r['judge']['matches_freeze'])
        self.assertIsNone(r['judge']['reliability'])
        self.assertIn('cli_version', r['judge']['differences'])

    def test_duplicate_case_ids_rejected(self):
        self.summaries.append(dict(self.summaries[0]))
        with self.assertRaises(ValueError):
            self.run_eval()

    def test_summaries_without_model_input_are_not_judged(self):
        self.summaries[1].pop('model_input')
        r = self.run_eval(caller=lambda ledger: FakeCaller(ledger))
        self.assertEqual((r['judge']['judged'], r['judge']['not_judged_no_model_input']), (1, 1))

    def test_changed_summaries_are_judged_fresh(self):
        self.run_eval(caller=lambda ledger: FakeCaller(ledger, output=judged('supported')))
        self.summaries[0]['prediction'] = 'The patient is 40 years old today.'
        r = self.run_eval(caller=lambda ledger: FakeCaller(ledger, output=judged('contradicted')))
        self.assertEqual(r['judge']['rates']['contradicted']['numerator'], 2)

    def test_preflight_reports_freeze_before_terms(self):
        from factuality.evaluate import judge_preflight
        with patch('factuality.evaluate.rubric_hash', return_value='edited'):
            with self.assertRaises(ValueError) as e:
                judge_preflight(bundle(), manifest={}, cli_version_now='x')
        self.assertIn('published freeze', str(e.exception))

    def test_counterfactual(self):
        b = bundle()
        b.cf_cases = [{'case_id': 'cf:age_shift:val:1', 'dialogue': D, 'requested_section': 'history of present illness',
                       'counterfactual': {'family': 'age_shift', 'gold': {'age': 40, 'original_age': 33}}}]
        r = evaluate_model('m', self.summaries[:1], b, work_dir=self.tmp, reliability=RELIABILITY,
                           cf_summaries=[{'case_id': 'cf:age_shift:val:1', 'prediction': PRED}])
        self.assertEqual(r['counterfactual']['families']['age_shift']['faithful'], 1)


if __name__ == '__main__':
    unittest.main()
