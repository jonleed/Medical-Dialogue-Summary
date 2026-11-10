"""Synthetic diagnostics only. No downloads, weights, API calls or real gold labels."""
import copy
import unittest

from factuality.checks import demographic_suggestions
from factuality.scoring import score, validate_annotation


def case(source='Patient: I am 42 years old.', prediction='The patient is 42 years old.'):
    return {'case_id': 'synthetic:1', 'dataset_kind': 'synthetic', 'dialogue': source,
            'prediction': prediction, 'model_input': {'text': source, 'truncated': False}}


def span(text, phrase):
    start = text.index(phrase)
    return {'start': start, 'end': start + len(phrase), 'text': phrase}


def annotation(c, label='supported', fact_type='age', kind='synthetic'):
    return {'case_id': c['case_id'], 'case_hash': __import__('factuality.io', fromlist=['case_hash']).case_hash(c),
            'reviewer': 'synthetic fixture', 'reviewer_kind': kind, 'status': 'complete',
            'claims_complete': True, 'expected_facts_complete': True,
            'expected_facts_frozen': True, 'expected_facts_blind': True,
            'expected_facts_reviewer_kind': kind, 'expected_facts_reviewer': 'synthetic fixture', 'expected_facts_frozen_at': '2026-10-17T00:00:00Z',
            'unresolved': [], 'claims': [{'id': 'c1', 'text': c['prediction'], 'fact_type': fact_type,
              'label': label, 'visibility': 'visible', 'rationale': 'Synthetic full-source inspection.',
              'generated_span': span(c['prediction'], c['prediction']),
              'source_spans': [] if label == 'unsupported' else [span(c['dialogue'], c['dialogue'])],
              'visible_spans': [] if label == 'unsupported' else [span(c['model_input']['text'], c['model_input']['text'])]}],
            'expected_facts': []}


class DemographicDiagnostics(unittest.TestCase):
    def test_explicit_age_correct_and_wrong(self):
        for age, label in [(42, 'supported'), (24, 'contradicted')]:
            c = case(prediction=f'The patient is {age} years old.')
            self.assertEqual(demographic_suggestions(c)[0]['proposed_label'], label)

    def test_missing_age_is_not_automatically_an_omission(self):
        self.assertEqual(demographic_suggestions(case(prediction='The patient feels well.')), [])

    def test_relatives_doses_durations_clinician_questions_are_not_patient_age(self):
        for source in ['Patient: My mother is 42 years old.', 'Patient: I take 42 mg daily.',
                       'Patient: It started 42 years ago.', 'Doctor: Are you 42 years old?',
                       'Doctor: I am 42 years old.', 'Patient: I am not 42 years old.',
                       'Patient: I was 42 years old when it began.']:
            self.assertEqual([s['proposed_label'] for s in demographic_suggestions(case(source=source))], ['unclear'], source)

    def test_relative_output_not_patient_age(self):
        self.assertEqual(demographic_suggestions(case(prediction='Her mother is 24 years old.')), [])

    def test_explicit_sex_contradiction(self):
        c = case(source='Doctor: The patient is female.', prediction='The patient is male.')
        s = demographic_suggestions(c)
        self.assertEqual((s[0]['fact_type'], s[0]['proposed_label']), ('sex', 'contradicted'))

    def test_missing_ambiguous_conflicting_identity(self):
        for source in ['Patient: My name is Mary.', 'Doctor: She feels well.',
                       'Doctor: The patient is female.\nDoctor: The patient is male.']:
            self.assertEqual([s['proposed_label'] for s in demographic_suggestions(case(source, 'The patient is male.'))], ['unclear'])

    def test_gender_identity_distinct_from_sex(self):
        c = case('Patient: I identify as a woman.', 'The patient is male.')
        self.assertEqual([s['proposed_label'] for s in demographic_suggestions(c)], ['unclear'])

    def test_no_medication_or_diagnosis_regex_judge(self):
        for prediction in ['The patient takes aspirin.', 'The patient has diabetes.', 'The patient denies pain.']:
            self.assertEqual(demographic_suggestions(case('Patient: I have pain.', prediction)), [])

    def test_truncated_source_not_used_as_visible_support(self):
        c = case(prediction='The patient is 24 years old.')
        c['model_input'] = {'text': 'Patient: Hello.', 'truncated': True}
        self.assertEqual(demographic_suggestions(c)[0]['visibility'], 'truncated')


class ScoringDiagnostics(unittest.TestCase):
    def test_pending_not_a_pass_and_na_zero_denominator(self):
        s = score([case()], [])
        self.assertEqual(s['human']['cases_with_annotations'], 0)
        self.assertIsNone(s['human']['full_source']['age_contradictions']['rate'])
        self.assertEqual(s['human']['passing_complete_cases'], 0)

    def test_age_contradiction_not_double_counted(self):
        c = case(prediction='The patient is 24 years old.')
        s = score([c], [annotation(c, 'contradicted')])['synthetic']['full_source']
        self.assertEqual(s['age_contradictions']['numerator'], 1)
        self.assertEqual(s['unsupported_claims']['numerator'], 0)
        self.assertEqual(s['summaries_with_error']['numerator'], 1)

    def test_invented_medication_diagnosis_and_negation_with_known_labels(self):
        for text, typ, label in [('Takes aspirin.', 'medication', 'unsupported'),
                                  ('Has diabetes.', 'diagnosis', 'unsupported'),
                                  ('Denies pain.', 'symptom', 'contradicted')]:
            c = case('Patient: I have pain.', text)
            s = score([c], [annotation(c, label, typ)])['synthetic']['full_source']
            self.assertEqual(s['summaries_with_error']['numerator'], 1)

    def test_omission_exclusion_and_truncation(self):
        c = case('Patient: I am 42 years old. I have pain.', 'The patient is 42 years old.')
        a = annotation(c)
        a['expected_facts'] = [{'id':'f1', 'text':'pain', 'fact_type':'symptom', 'label':'omitted',
             'visibility':'truncated', 'source_spans':[span(c['dialogue'], 'I have pain.')],
             'visible_spans':[], 'rationale':'Synthetic fact beyond token limit.'},
            {'id':'f2', 'text':'family detail', 'fact_type':'other', 'label':'excluded',
             'visibility':'unclear', 'source_spans':[], 'visible_spans':[], 'rationale':'Out of section.'}]
        s = score([c], [a])['synthetic']
        self.assertEqual(s['full_source']['expected_fact_omissions']['denominator'], 1)
        self.assertEqual(s['model_visible']['expected_fact_omissions']['denominator'], 0)
        self.assertEqual(s['truncated_expected_facts'], 1)

    def test_unclear_partial_no_pass(self):
        c = case(); a = annotation(c, 'unclear'); a['status'] = 'partial'
        s = score([c], [a])['synthetic']
        self.assertEqual(s['passing_complete_cases'], 0)
        self.assertEqual(s['unclear_claims'], 1)

    def test_agent_review_never_human(self):
        c = case(); c['dataset_kind'] = 'real'
        s = score([c], [annotation(c, kind='agent')])
        self.assertEqual(s['human']['cases_with_annotations'], 0)
        self.assertEqual(s['agent']['cases_with_annotations'], 1)

    def test_reject_stale_duplicate_and_invalid_spans(self):
        c = case(); a = annotation(c)
        for mutate in [lambda x: x.update(case_hash='stale'),
                       lambda x: x['claims'][0]['generated_span'].update(text='bad'),
                       lambda x: x.update(reviewer=''),
                       lambda x: x.update(claims=[]),
                       lambda x: x.update(expected_facts_blind=False),
                       lambda x: x['claims'][0].update(visible_spans=[])]:
            bad = copy.deepcopy(a); mutate(bad)
            with self.assertRaises(ValueError): validate_annotation(c, bad)
        with self.assertRaises(ValueError): score([c], [a, a])

    def test_reject_mixed_empirical_synthetic_denominators(self):
        real = case(); real['case_id'] = 'val:1'; real['dataset_kind'] = 'real'
        with self.assertRaises(ValueError): score([case(), real], [])


if __name__ == '__main__': unittest.main()
