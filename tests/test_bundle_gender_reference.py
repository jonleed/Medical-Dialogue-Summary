"""The published reviewer labels and the rule checker apply the same gender-reference rule."""
import unittest
from pathlib import Path

from factuality.demographics import GENDER_REFERENCE
from factuality.rebuild import load_bundle
from factuality.rules import suggestions_v3

BUNDLE = Path(__file__).resolve().parent.parent / 'data' / 'factuality_benchmark'
DETERMINATE = {'supported', 'contradicted'}


def _overlap(a, b):
    return a['start'] < b['end'] and b['start'] < a['end']


class BundleGenderReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bundle = load_bundle(BUNDLE)
        cases = {(m, c['case_id']): c for m, cs in bundle.summaries.items() for c in cs}
        cls.pairs = []
        for label in bundle.labels:
            key = (label['model'], label['case_id'])
            suggestions = [s for s in suggestions_v3(cases[key]) if s['fact_type'] in GENDER_REFERENCE]
            for claim in label['claims']:
                if claim['fact_type'] not in GENDER_REFERENCE or claim.get('subject', 'patient') != 'patient':
                    continue
                match = next((s for s in suggestions if _overlap(s['generated_span'], claim['generated_span'])), None)
                cls.pairs.append((key, claim, match))

    def test_labels_are_not_opposite_to_the_checker(self):
        clashes = [(k, c['id'], c['label'], m['proposed_label']) for k, c, m in self.pairs
                   if m and {c['label'], m['proposed_label']} == DETERMINATE]
        self.assertEqual(clashes, [])

    def test_claims_with_no_evidence_are_unsupported_or_unclear(self):
        # A claim the checker can find no cue for must never be labelled supported or contradicted
        # unless the reviewer cites evidence the checker does not model (the checker abstains, never asserts).
        for key, claim, match in self.pairs:
            if claim['label'] in DETERMINATE:
                self.assertTrue(claim['source_spans'], (key, claim['id']))
            if match and match['proposed_label'] in DETERMINATE:
                self.assertIn(claim['label'], DETERMINATE | {'unclear'}, (key, claim['id']))

    def test_unsupported_claims_have_no_cue_anywhere(self):
        for key, claim, match in self.pairs:
            if claim['label'] == 'unsupported':
                self.assertEqual(claim['source_spans'], [], (key, claim['id']))
                self.assertTrue(match is None or match['proposed_label'] == 'unclear', (key, claim['id']))


if __name__ == '__main__':
    unittest.main()
