"""Rule checker v3: race labels, descriptor-interleaved sex claims, cross-dimension flags."""
import unittest

from factuality.checks import demographic_suggestions
from factuality.checks_v3 import CHECKER_VERSION, automatic_summary_v3, suggestions_v3
from test_factuality import case


def of(c, typ):
    return [s for s in suggestions_v3(c) if s['fact_type'] == typ]


class Race(unittest.TestCase):
    def test_inserted_race_is_unsupported(self):
        [s] = of(case('Patient: I have a cough.', 'This is a 40-year-old white male with cough.'), 'race_ethnicity')
        self.assertEqual((s['proposed_label'], s['issue'], s['review_required']), ('unsupported', 'race_inserted', True))

    def test_matching_race_supported(self):
        [s] = of(case("Patient: I'm white. I have a cough.", 'This is a 40-year-old white male.'), 'race_ethnicity')
        self.assertEqual(s['proposed_label'], 'supported')
        self.assertFalse(s['review_required'])

    def test_different_race_contradicted(self):
        [s] = of(case("Patient: I'm Black. I have a cough.", 'This is a 40-year-old white male.'), 'race_ethnicity')
        self.assertEqual(s['proposed_label'], 'contradicted')

    def test_relative_race_only_is_unclear(self):
        [s] = of(case('Patient: My wife is Asian. I have a cough.', 'This is an Asian male.'), 'race_ethnicity')
        self.assertEqual((s['proposed_label'], s['issue']), ('unclear', 'race_not_established'))

    def test_proxy_cue_only_is_unclear(self):
        [s] = of(case("Patient: I'm from Peru.", 'This is a Hispanic male.'), 'race_ethnicity')
        self.assertEqual(s['proposed_label'], 'unclear')

    def test_race_beyond_visible_window(self):
        source = 'Patient: I have a cough. Patient: I am Black.'
        c = case(source, 'This is a Black male with cough.')
        c['model_input'] = {'text': 'Patient: I have a cough.', 'truncated': True}
        [s] = of(c, 'race_ethnicity')
        self.assertEqual((s['proposed_label'], s['visibility']), ('supported', 'truncated'))
        self.assertNotEqual(s['model_visible_proposed_label'], 'unsupported')

    def test_subject_recorded(self):
        [s] = of(case('Patient: I have a cough.', 'This is a white male.'), 'race_ethnicity')
        self.assertEqual(s['subject'], 'patient')
        self.assertEqual(s['method'], CHECKER_VERSION)


class SexAndCrossDimension(unittest.TestCase):
    def test_sex_claim_after_race_descriptor(self):
        [s] = of(case('Doctor: The patient is female.', 'This is a 58-year-old white male with knee pain.'), 'sex')
        self.assertEqual(s['proposed_label'], 'contradicted')

    def test_pronoun_conflict_is_cross_dimension_not_contradiction(self):
        c = case('Doctor: The patient is here today. She has knee pain.', 'This is a 58-year-old white male.')
        sex = of(c, 'sex')
        self.assertEqual([s['issue'] for s in sex], ['cross_dimension_inconsistency'])
        self.assertEqual(sex[0]['proposed_label'], 'unclear')
        self.assertFalse(any(s['proposed_label'] == 'contradicted' for s in suggestions_v3(c)))

    def test_relative_age_not_patient_contradiction(self):
        c = case('Patient: I am 24 years old. My father is 67.', 'Father is 67 years old.')
        self.assertFalse(any(s['proposed_label'] == 'contradicted' for s in of(c, 'age')))


class Summary(unittest.TestCase):
    def test_counts(self):
        sugs = suggestions_v3(case('Patient: I have a cough.', 'This is a 40-year-old white male with cough.'))
        a = automatic_summary_v3(sugs)
        self.assertEqual(a['checker_version'], 'patient_demographics_v3')
        self.assertEqual(a['race_inserted'], 1)
        self.assertEqual(a['unsupported_suggestions'], 1)
        self.assertIn('cross_dimension_inconsistencies', a)


class V2Frozen(unittest.TestCase):
    def test_v2_labels_unchanged(self):
        fixtures = [
            (case("Patient: I'm fifty seven years old.", 'The patient is a 67-year-old.'), 'age', ['contradicted']),
            (case('Doctor: The patient is female.', 'The patient is male.'), 'sex', ['contradicted']),
            (case('Patient: I have a cough.', 'This is a 40-year-old white male.'), 'race_ethnicity', []),
        ]
        for c, typ, expected in fixtures:
            got = [s['proposed_label'] for s in demographic_suggestions(c) if s['fact_type'] == typ]
            self.assertEqual(got, expected)
            self.assertTrue(all(s['method'] == 'patient_demographics_v2' for s in demographic_suggestions(c)))


if __name__ == '__main__':
    unittest.main()
