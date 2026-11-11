"""Offline synthetic and real saved-output regressions for demographic detection."""
import unittest
from pathlib import Path
from factuality.checks import demographic_suggestions
from factuality.io import read_jsonl
from test_factuality import case


def items(c, typ):
    return [s for s in demographic_suggestions(c) if s['fact_type']==typ]


def errors(c, typ):
    return [s for s in items(c,typ) if s['proposed_label']=='contradicted']


class AgeV2Diagnostics(unittest.TestCase):
    def test_words_units_and_summary_introductions(self):
        for source,out in [
          ("Patient: I'm fifty seven years old.", 'The patient is a 67-year-old.'),
          ('Patient: I am seventy one years of age.', 'This is a 72-year-old patient.'),
          ('Patient: I am thirty-nine years old.', 'The patient is a 38-year-old.')]:
            self.assertEqual(len(errors(case(source,out),'age')),1)

    def test_age_question_answer_and_confirmation(self):
        for source in ["Doctor: How old are you now?\nPatient: I'm thirty nine.",
                       'Doctor: Can you tell me your age?\nPatient: I am thirty nine.',
                       'Doctor: Just to confirm, you are thirty nine years old?\nPatient: Yes.',
                       'Doctor: Let me start by getting your age.\nGuest_family: She is thirty nine!']:
            self.assertEqual(len(errors(case(source,'This is a 40-year-old patient.'),'age')),1,source)

    def test_self_age_inside_question_and_birthday(self):
        for source in ['Patient: Is that common for a fifty one year old woman like me?',
                       "Doctor: Happy birthday! How young are you now?\nPatient: I just turned fifty one."]:
            self.assertEqual(len(errors(case(source,'This is a 50-year-old patient.'),'age')),1)

    def test_correct_paraphrase(self):
        s=items(case("Doctor: How old are you?\nPatient: I'm seventy three.",'The patient is a 73-year-old.'),'age')
        self.assertEqual(s[0]['proposed_label'],'supported')

    def test_other_numbers_and_unconfirmed_questions_do_not_establish_age(self):
        for source in ['Patient: My father is fifty seven years old.',
          'Patient: I take fifty seven mg daily.', 'Patient: It began fifty seven years ago.',
          'Doctor: How old was your father?\nPatient: He was fifty seven.',
          'Doctor: Are you fifty seven years old?', 'Doctor: I am fifty seven years old.',
          'Patient: I was fifty seven when it began.', 'Patient: I am not fifty seven years old.',
          'Doctor: Are you fifty seven years old?\nPatient: No.',
          'Patient: My food journal uses fifty seven calorie points.']:
            self.assertFalse(errors(case(source,'The patient is a 67-year-old.'),'age'),source)

    def test_relative_output_not_patient_age(self):
        for out in ['Her father is a 67-year-old man.','His son is 67 years old.','Mother died at age 67.']:
            self.assertFalse(items(case('Patient: I am fifty seven years old.',out),'age'))

    def test_unestablished_and_conflicting_ages_are_review_candidates(self):
        for source in ['Patient: I have pain.',
          'Patient: I am fifty seven years old.\nPatient: I am sixty seven years old.']:
            suggestions=items(case(source,'The patient is a 67-year-old.'),'age')
            self.assertEqual(suggestions[0]['proposed_label'],'unclear')
            self.assertTrue(suggestions[0]['review_required'])

    def test_dialogue_like_output_multiple_age_assertions(self):
        c=case("Patient: I'm fifty seven years old.","Patient: I'm fifty seven years old. Patient: I'm seventy seven years old and I'm twenty five years old now.")
        self.assertEqual(len(errors(c,'age')),2)


class PronounV2Diagnostics(unittest.TestCase):
    def test_patient_companion_and_direct_reference(self):
        for source in ["Doctor: Are you the husband?\nGuest_family: I wanted to be here with her today.",
          'Doctor: The patient is here. She has pain.',
          "Guest_clinician: This is the patient. She's here for follow-up."]:
            self.assertEqual(len(errors(case(source,'The patient said that he has pain.'),'pronoun')),1,source)

    def test_truncated_patient_reference_not_borrowed_from_surgeon_or_nephew(self):
        visible='Patient: Doctor Q operated on me. She is a good doctor.\nDoctor: Your nephew called and he lost the paper.'
        source=visible+'\nDoctor: Are you the husband?\nGuest_family: I wanted to be here with her today.'
        c=case(source,'The patient said that he has pain.')
        c['model_input']={'text':visible,'truncated':True}
        suggestion=errors(c,'pronoun')[0]
        self.assertEqual(suggestion['visibility'],'truncated')
        self.assertEqual(suggestion['visible_spans'],[])
        self.assertIsNone(suggestion['model_visible_proposed_label'])

    def test_relatives_clinicians_and_names_not_patient_identity(self):
        for source in ['Patient: My mother is here. She feels well.',
          'Patient: My doctor operated on me. She is helpful.',
          'Doctor: Your psychiatrist wrote a report.\nPatient: Okay.\nDoctor: He thinks you are well.',
          'Patient: My name is Mary.', 'Patient: I live with my husband.',
          'Doctor: She feels well.']:
            self.assertFalse(errors(case(source,'The patient says he has pain.'),'pronoun'),source)

    def test_honorific_mismatch_only_unclear(self):
        c=case('Doctor: Hello, miss.\nPatient: I have a rash.','He has a rash.')
        s=items(c,'pronoun')[0]
        self.assertEqual(s['proposed_label'],'unclear')
        self.assertEqual(s['issue'],'gendered_address_inconsistency')

    def test_conflicting_patient_pronouns_are_unclear(self):
        c=case('Doctor: The patient is here. She has pain.\nDoctor: The patient says he has pain.','He has pain.')
        self.assertEqual(items(c,'pronoun')[0]['proposed_label'],'unclear')

    def test_other_person_output_does_not_inherit_patient_pronoun(self):
        c=case('Doctor: The patient is here. She has pain.',"The patient's father is here. He has diabetes.")
        self.assertFalse(errors(c,'pronoun'))


class AttributionSafetyRegressions(unittest.TestCase):
    def test_provider_and_possessive_relative_are_not_patient_evidence(self):
        for source in ['Doctor: She examined the patient and prescribed antibiotics.',
                       "Doctor: The patient's wife says she is worried.",
                       'Doctor: The patient saw a nurse. She recommended rest.',
                       'Doctor: The patient saw a physician. He recommended rest.',
                       'The patient saw a nurse. She recommended rest.',
                       "The patient's wife says she is worried."]:
            s=items(case(source,'She reports pain.'),'pronoun')
            self.assertEqual(s[0]['proposed_label'],'unclear')
        c=case('Doctor: The patient is here. He has pain.',"The patient's wife says she is worried.")
        self.assertFalse(errors(c,'pronoun'))
        c=case('Doctor: The patient is here. He has pain.','She examined the patient and recommended rest.')
        self.assertFalse(errors(c,'pronoun'))

    def test_quoted_or_qualified_demographics_not_definite_evidence(self):
        for source,out,typ in [
          ("Patient: My mother said, 'I am fifty seven years old.'",'The patient is a 57-year-old.','age'),
          ('Patient: I think I am fifty seven years old.','The patient is a 57-year-old.','age'),
          ('Doctor: It is not true that the patient is male.','The patient is male.','sex'),
          ('Patient: If I am a man, will the treatment differ?','The patient identifies as a man.','gender'),
          ('Patient: I am fifty seven years old in that old photo.','The patient is a 57-year-old.','age')]:
            self.assertEqual(items(case(source,out),typ)[0]['proposed_label'],'unclear',source)

    def test_complete_pronoun_anchor_required_for_visibility(self):
        c=case('Doctor: She is the patient.','He reports pain.')
        c['model_input']={'text':'Doctor: She','truncated':True}
        suggestion=items(c,'pronoun')[0]
        self.assertEqual(suggestion['proposed_label'],'contradicted')
        self.assertEqual(suggestion['visibility'],'truncated')
        self.assertIsNone(suggestion['model_visible_proposed_label'])

    def test_nonbinary_spelling_is_equivalent(self):
        c=case('Patient: I am nonbinary.','The patient identifies as non-binary.')
        self.assertEqual(items(c,'gender')[0]['proposed_label'],'supported')

    def test_compound_descriptions_do_not_establish_standalone_sex(self):
        for source in ['Patient: I am male-to-female transgender.',
                       'Patient: I am female-to-male transgender.']:
            for out in ['The patient is male.','The patient is female.']:
                self.assertEqual(items(case(source,out),'sex')[0]['proposed_label'],'unclear')


class AuditIntegrationRegressions(unittest.TestCase):
    def test_review_flags_are_visible_and_never_become_human_labels(self):
        import tempfile
        from factuality.review import write_queue
        from factuality.scoring import score
        c=case("Patient: I'm fifty seven years old.",'The patient is a 67-year-old.')
        c.update(requested_section='HPI',reference='',dataset_kind='real')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'queue'
            summary=write_queue([c],path)
            self.assertEqual(summary['contradiction_suggestions'],1)
            self.assertIn('67-year-old',(path/'review.html').read_text())
            self.assertIn('Automatic suggestions',(path/'review.html').read_text())
            self.assertEqual(read_jsonl(path/'annotations.jsonl')[0]['claims'],[])
            metrics=score([c],read_jsonl(path/'annotations.jsonl'))
            self.assertEqual(metrics['human']['fully_reviewed_cases'],0)
            self.assertIsNone(metrics['human']['full_source']['age_contradictions']['rate'])

    def test_review_candidates_count_separately_from_contradictions(self):
        from factuality.checks import automatic_summary
        suggestions=demographic_suggestions(case('Patient: I have pain.','The patient is a 67-year-old.'))
        summary=automatic_summary(suggestions)
        self.assertEqual(summary['unclear_suggestions'],1)
        self.assertEqual(summary['flagged_cases'],1)
        self.assertEqual(summary['contradiction_flagged_cases'],0)

class SavedOutputRegressions(unittest.TestCase):
    def test_real_source_age_misses_and_truncation(self):
        path=Path('results/factuality/local/biobart-flask-2026-10-17/cases.jsonl')
        if not path.exists():self.skipTest('Original private/local run exports unavailable; synthetic regressions still run')
        rows={c['case_id']:c for c in read_jsonl(path)}
        for cid in ['val:9','val:24']:
            s=errors(rows[cid],'age')
            self.assertEqual(len(s),1,cid)
            self.assertEqual(s[0]['visibility'],'visible')
        s=errors(rows['val:37'],'pronoun')
        self.assertEqual(len(s),1)
        self.assertEqual(s[0]['visibility'],'truncated')
        self.assertEqual(items(rows['val:66'],'pronoun')[0]['issue'],'gendered_address_inconsistency')


if __name__=='__main__':unittest.main()
