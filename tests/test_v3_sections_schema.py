"""v3 canonical section keys and race/subject annotation schema."""
import unittest

from factuality.scoring import validate_annotation
from factuality.sections import canonical_section, with_canonical_sections
from test_factuality import annotation, case


class Sections(unittest.TestCase):
    def test_case_variants_collapse(self):
        a = {'requested_section': 'History of Present Illness'}
        b = {'requested_section': 'history  of present illness'}
        self.assertEqual(canonical_section(a), canonical_section(b))
        self.assertEqual(canonical_section(a), 'history of present illness')

    def test_prompt_untouched_and_input_not_mutated(self):
        c = {'requested_section': 'History of Present Illness', 'prompt': 'P'}
        out = with_canonical_sections([c])[0]
        self.assertEqual(out['requested_section'], 'History of Present Illness')
        self.assertEqual(out['prompt'], 'P')
        self.assertEqual(out['canonical_section'], 'history of present illness')
        self.assertNotIn('canonical_section', c)


class SchemaV3(unittest.TestCase):
    def _race(self, subject):
        c = case("Patient: I'm white.", 'The patient is white.')
        a = annotation(c, label='supported', fact_type='race_ethnicity')
        a['claims'][0]['subject'] = subject
        return c, a

    def test_race_type_and_subject_accepted(self):
        validate_annotation(*self._race('patient'))

    def test_unknown_subject_rejected(self):
        with self.assertRaises(ValueError):
            validate_annotation(*self._race('neighbor'))


if __name__ == '__main__':
    unittest.main()
