"""Offline rebuild of the published aggregates from a benchmark bundle."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from factuality.rebuild import OUTPUTS, load_bundle, rebuild

D = 'Doctor: How old are you? Patient: I am 40 years old. I have a cough.'
P = 'The patient is 41 years old with a cough.'


def span(text, phrase):
    s = text.index(phrase)
    return {'start': s, 'end': s + len(phrase), 'text': phrase}


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def make_bundle(root, optional=True):
    b = root / 'bundle'
    b.mkdir()
    cases = [{'case_id': f'val:{i}', 'dialogue': D, 'requested_section': 'history of present illness',
              'reference': 'ref', 'source': {'split': 'val'}} for i in (1, 2)]
    write_jsonl(b / 'cases.jsonl', cases)
    write_jsonl(b / 'summaries.jsonl', [{'model': 'm', 'case_id': c['case_id'], 'prediction': P,
                                         'model_input': {'text': D, 'truncated': False}} for c in cases])
    cf = {'case_id': 'cf:age_shift:val:1', 'dialogue': D, 'requested_section': 'history of present illness',
          'counterfactual': {'family': 'age_shift', 'gold': {'age': 41, 'original_age': 40}}}
    write_jsonl(b / 'counterfactual_cases.jsonl', [cf])
    write_jsonl(b / 'counterfactual_summaries.jsonl', [{'model': 'm', 'case_id': cf['case_id'], 'prediction': P,
                                                        'model_input': {'text': D, 'truncated': False}}])
    write_jsonl(b / 'plans.jsonl', [])
    claim = {'id': 'c1', 'text': 'Age 41', 'fact_type': 'age', 'subject': 'patient', 'label': 'contradicted',
             'visibility': 'visible', 'rationale': 'says 40', 'generated_span': span(P, '41 years old'),
             'source_spans': [span(D, '40 years old')], 'visible_spans': [span(D, '40 years old')]}
    write_jsonl(b / 'labels.jsonl', [{'case_id': c['case_id'], 'model': 'm', 'claims': [claim],
                                      'expected_facts': [{'id': 'f1', 'label': 'omitted', 'source_spans': [span(D, 'cough')]}]}
                                     for c in cases])
    (b / 'splits.json').write_text(json.dumps({'calibration': ['val:1'], 'heldout': ['val:2'], 'random': ['val:1', 'val:2'],
                                               'top_up': [], 'enrichment': []}))
    result = {'claims_complete': True, 'claims': [dict(claim, label='contradicted', evidence_quotes=['40 years old'],
                                                       generated_quote='41 years old')],
              'section_inventory': [{'text': 'cough', 'source_quote': 'cough', 'status': 'omitted', 'generated_quote': '',
                                     'source_span': span(D, 'cough'), 'generated_span': None}]}
    rows = [{'model': 'm', 'case_id': 'val:1', 'split': 'calibration-r2', 'trial': 1, 'status': 'complete', 'result': result},
            {'model': 'm', 'case_id': 'val:2', 'split': 'heldout', 'trial': 1, 'status': 'complete', 'result': result},
            {'model': 'm', 'case_id': 'val:2', 'split': 'heldout', 'trial': 2, 'status': 'complete', 'result': result},
            {'model': 'm', 'case_id': 'cf:age_shift:val:1', 'split': 'counterfactual', 'trial': 1, 'status': 'complete', 'result': result}]
    write_jsonl(b / 'judge_outputs.jsonl', rows)
    (b / 'judge_config.json').write_text('{}')
    (b / 'DATASHEET.md').write_text('x')
    if optional:
        (b / 'rubric_revisions.md').write_text('x')
    return b


class Rebuild(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_writes_outputs_deterministically(self):
        b = make_bundle(self.root)
        first, second = rebuild(b, self.root / 'o1', resamples=50), rebuild(b, self.root / 'o2', resamples=50)
        self.assertEqual(sorted(first), sorted(OUTPUTS))
        for name in OUTPUTS:
            self.assertEqual(first[name].read_bytes(), second[name].read_bytes())

    def test_content(self):
        out = rebuild(make_bundle(self.root), self.root / 'o', resamples=50)
        m = json.loads(out['metrics.json'].read_text())
        self.assertEqual(m['heldout']['claude']['age_contradiction']['positives'], 1)
        self.assertEqual(json.loads(out['counterfactual.json'].read_text())['m']['families']['age_shift']['faithful'], 1)
        self.assertEqual(json.loads(out['screening.json'].read_text())['models'][0]['model'], 'm')
        self.assertIn('definition', json.loads(out['demographic_error_rates.json'].read_text()))

    def test_optional_files_absent(self):
        out = rebuild(make_bundle(self.root, optional=False), self.root / 'o', resamples=50)
        self.assertEqual(sorted(out), sorted(OUTPUTS))

    def test_load_bundle_joins_summaries_with_cases(self):
        b = load_bundle(make_bundle(self.root))
        self.assertEqual(b.summaries['m'][0]['dialogue'], D)
        self.assertEqual(b.summaries['m'][0]['prediction'], P)
        self.assertIn(('m', 'val:2'), b.judge['heldout'])
        self.assertEqual(set(b.judge_trials), {'t1', 't2'})


class Check(unittest.TestCase):
    def test_check_fails_on_difference(self):
        root = Path(tempfile.mkdtemp())
        b = make_bundle(root)
        pub = root / 'pub'
        rebuild(b, pub, resamples=50)
        (pub / 'metrics.json').write_text('{}\n')
        from factuality.rebuild import check
        self.assertEqual(check(b, pub, resamples=50), ['metrics.json'])
        rebuild(b, root / 'pub2', resamples=50)
        self.assertEqual(check(b, root / 'pub2', resamples=50), [])


if __name__ == '__main__':
    unittest.main()


class Published(unittest.TestCase):
    def test_only_rebuildable_outputs_are_published(self):
        tracked = subprocess.run(['git', 'ls-files', 'results/factuality'], capture_output=True, text=True,
                                 check=True).stdout.split()
        self.assertEqual(sorted(tracked), sorted(f'results/factuality/{n}' for n in OUTPUTS))
