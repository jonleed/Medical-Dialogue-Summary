"""v3 output-validity gate: non-summaries never enter factuality denominators."""
import unittest

from factuality.validity import classify_output, validity_summary

DIALOGUE = ('Doctor: How are you feeling today? Patient: I have had a cough for two weeks '
            'and some fever at night. Doctor: Any shortness of breath? Patient: No, none at all.')


class Validity(unittest.TestCase):
    def test_prompt_prefix_is_echo(self):
        out = 'Summarize the following doctor-patient dialogue into a detailed plan clinical note: ' + DIALOGUE
        self.assertEqual(classify_output(out, DIALOGUE), 'prompt_echo')

    def test_verbatim_copy_is_echo(self):
        self.assertEqual(classify_output(DIALOGUE, DIALOGUE), 'prompt_echo')

    def test_label_only_is_empty(self):
        self.assertEqual(classify_output('history of present illness:   ', DIALOGUE), 'empty')

    def test_none_is_empty(self):
        self.assertEqual(classify_output(None, DIALOGUE), 'empty')

    def test_repetition_is_degenerate(self):
        self.assertEqual(classify_output('plan: ' + 'patient is fine ' * 20, DIALOGUE), 'degenerate')

    def test_real_summary(self):
        out = 'history of present illness: The patient reports a two-week cough with nightly fever and denies dyspnea.'
        self.assertEqual(classify_output(out, DIALOGUE), 'summary')

    def test_short_extractive_summary_is_not_echo(self):
        dialogue = 'Patient: He is up to date on his immunizations, as far as I know.'
        self.assertEqual(classify_output('history of present illness: He is up to date on his immunizations.', dialogue), 'summary')

    def test_short_list_is_not_degenerate(self):
        self.assertEqual(classify_output('allergies: No known drug allergies.', DIALOGUE), 'summary')
        self.assertEqual(classify_output('chief complaint: Abdominal pain, diarrhea, vomiting.', DIALOGUE), 'summary')

    def test_short_summary_is_not_echo(self):
        self.assertEqual(classify_output('chief complaint: Cough.', DIALOGUE), 'summary')

    def test_summary_counts(self):
        cases = [{'case_id': 'a', 'dialogue': DIALOGUE, 'prediction': DIALOGUE},
                 {'case_id': 'b', 'dialogue': DIALOGUE, 'prediction': 'plan: Rest and fluids.'}]
        s = validity_summary(cases)
        self.assertEqual(s['counts']['prompt_echo'], 1)
        self.assertEqual(s['counts']['summary'], 1)
        self.assertEqual(s['rate_summary'], 0.5)
        self.assertEqual(s['per_case'], {'a': 'prompt_echo', 'b': 'summary'})


if __name__ == '__main__':
    unittest.main()
