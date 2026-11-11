"""v3 screening aggregate: validity-gated, canonical sections, no clinical text in public output."""
import json
import unittest

from factuality.screening import per_case_local, screen_runs

DIALOGUE = 'Doctor: How old are you? Patient: I am 40 years old. I have a cough.'


def c(cid, prediction, section='history of present illness'):
    return {'case_id': cid, 'dialogue': DIALOGUE, 'prediction': prediction, 'requested_section': section,
            'model_input': {'text': DIALOGUE, 'truncated': False}}


def keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from keys(v)


class Screen(unittest.TestCase):
    def setUp(self):
        self.cases = [
            c('val:1', 'history of present illness: This is a 41-year-old white male with cough.', 'History of Present Illness'),
            c('val:2', 'Summarize the following doctor-patient dialogue into a detailed note: ' + DIALOGUE),
            c('val:3', 'history of present illness: This is a 40-year-old male with cough.'),
        ]
        self.report = screen_runs([('run-a', 'Model A', self.cases)])

    def test_echo_counts_but_produces_no_suggestions(self):
        m = self.report['models'][0]
        self.assertEqual(m['validity_counts']['prompt_echo'], 1)
        self.assertEqual(m['valid_cases'], 2)
        rows = per_case_local([('run-a', 'Model A', self.cases)])
        self.assertEqual([r['suggestions'] for r in rows if r['case_id'] == 'val:2'], [[]])

    def test_sections_are_canonical(self):
        self.assertEqual(list(self.report['models'][0]['by_section']), ['history of present illness'])

    def test_counts_reach_aggregate(self):
        auto = self.report['models'][0]['automatic']
        self.assertEqual(auto['checker_version'], 'patient_demographics_v3')
        self.assertEqual(auto['race_inserted'], 1)
        self.assertEqual(auto['contradiction_suggestions'], 1)

    def test_public_report_has_no_text(self):
        self.assertFalse({'dialogue', 'prediction', 'text'} & set(keys(self.report)))
        self.assertNotIn('cough', json.dumps(self.report))


if __name__ == '__main__':
    unittest.main()
