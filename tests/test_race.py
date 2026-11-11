"""v3 race/ethnicity extraction: patient-anchored, explicit-only, with clinical-term exclusions."""
import unittest

from factuality.race import RACE_TERMS, has_race_vocabulary, proxy_cues, race_claims, race_evidence


def values(facts):
    return [f['value'] for f in facts]


class Vocabulary(unittest.TestCase):
    def test_eight_normalized_groups(self):
        self.assertEqual(set(RACE_TERMS.values()), {
            'white', 'black', 'hispanic_latino', 'asian', 'american_indian_alaska_native',
            'nhpi', 'mena', 'multiracial'})


class Claims(unittest.TestCase):
    def test_age_race_sex_intro_after_section_label(self):
        text = 'history of present illness: This is a 58-year-old white female with knee pain.'
        claims = race_claims(text)
        self.assertEqual(values(claims), ['white'])
        self.assertEqual(claims[0]['span']['text'], 'white')

    def test_multiword_term(self):
        self.assertEqual(values(race_claims('The patient is a 62-year-old African American who has diabetes.')), ['black'])

    def test_plain_patient_statement(self):
        self.assertEqual(values(race_claims('The patient is Hispanic.')), ['hispanic_latino'])

    def test_bare_intro_without_patient_phrase(self):
        self.assertEqual(values(race_claims('A 40-year-old Caucasian male presents with chest pain.')), ['white'])

    def test_clinical_white_terms_excluded(self):
        self.assertEqual(race_claims('Labs: white blood cell count 12.'), [])
        self.assertEqual(race_claims('WBC elevated, white count normal.'), [])

    def test_clinical_black_terms_excluded(self):
        self.assertEqual(race_claims('She reports black stools and a blackout.'), [])

    def test_relative_race_excluded(self):
        self.assertEqual(race_claims('Her mother is Hispanic.'), [])


class Evidence(unittest.TestCase):
    def test_self_statement_curly_apostrophe(self):
        self.assertEqual(values(race_evidence('Patient: I’m Black.')), ['black'])

    def test_question_answer(self):
        facts = race_evidence('Doctor: What is your race or ethnicity? Patient: Asian.')
        self.assertEqual(values(facts), ['asian'])
        self.assertEqual(facts[0]['basis'], 'race_question_answer')

    def test_clinician_statement(self):
        self.assertEqual(values(race_evidence('Doctor: The patient is a Caucasian male.')), ['white'])

    def test_self_statement_with_age_between(self):
        self.assertEqual(values(race_evidence('Patient: I am forty one year old Hispanic.')), ['hispanic_latino'])

    def test_confirmed_clinician_question(self):
        facts = race_evidence('Doctor: You identify as Asian, is that right? \nPatient: Yes, that is right.')
        self.assertEqual(values(facts), ['asian'])
        self.assertEqual(facts[0]['basis'], 'confirmed_race_question')

    def test_unconfirmed_clinician_question(self):
        self.assertEqual(race_evidence('Doctor: You identify as Asian, is that right? \nPatient: No.'), [])

    def test_relative_not_patient_evidence(self):
        self.assertEqual(race_evidence('Patient: My husband is Native American.'), [])

    def test_nationality_is_proxy_not_race(self):
        text = "Patient: I'm from Mexico."
        self.assertEqual(race_evidence(text), [])
        self.assertTrue(proxy_cues(text))

    def test_vocabulary_scan(self):
        self.assertFalse(has_race_vocabulary('Patient: My white blood cells were high.'))
        self.assertTrue(has_race_vocabulary('Patient: My husband is Native American.'))
        self.assertFalse(has_race_vocabulary('Patient: I have a cough.'))


if __name__ == '__main__':
    unittest.main()
