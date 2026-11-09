"""Counterfactual packet: replayable edits, visible-window placement, construction gold."""
import unittest
from collections import Counter

from factuality.counterfactual import (GROUPS, apply_edits, build_packet, inject_race, remove_race,
                                       render_number, score_counterfactual, shift_age, swap_pronouns)
from factuality.generate import validate_case_request
from factuality.io import digest
from factuality.race import has_race_vocabulary, race_evidence

SECTION = 'history of present illness'


def base(dialogue, cid='val:1'):
    return {'schema_version': 1, 'dataset_kind': 'real', 'case_id': cid, 'protocol': 'flask-note',
            'source': {'split': 'val', 'dialogue_id': cid[4:], 'dialogue_sha256': digest(dialogue)},
            'dialogue': dialogue, 'requested_section': SECTION,
            'prompt': f'Summarize the following doctor-patient dialogue into a detailed {SECTION} clinical note: {dialogue}',
            'prediction': 'old output', 'model_input': {'text': dialogue, 'truncated': False},
            'generation_status': 'complete'}


def words(n):
    return lambda s: len(s.split())


class Edits(unittest.TestCase):
    def test_apply_edits(self):
        self.assertEqual(apply_edits('abcdef', [{'start': 1, 'end': 3, 'old': 'bc', 'new': 'XY'}]), 'aXYdef')

    def test_apply_edits_right_to_left(self):
        edits = [{'start': 0, 'end': 1, 'old': 'a', 'new': 'AA'}, {'start': 4, 'end': 5, 'old': 'e', 'new': 'E'}]
        self.assertEqual(apply_edits('abcdef', edits), 'AAbcdEf')

    def test_render_number(self):
        self.assertEqual(render_number(64, 'fifty seven'), 'sixty four')
        self.assertEqual(render_number(64, 'fifty-seven'), 'sixty-four')
        self.assertEqual(render_number(64, '57'), '64')
        self.assertEqual(render_number(70, 'sixty three'), 'seventy')


class Families(unittest.TestCase):
    def assert_replayable(self, b, cf):
        self.assertEqual(apply_edits(b['dialogue'], cf['counterfactual']['edits']), cf['dialogue'])
        validate_case_request(cf)
        self.assertIsNone(cf['prediction'])
        self.assertIsNone(cf['model_input'])

    def test_inject_race(self):
        b = base('Doctor: What brings you in? Patient: I have a cough.')
        cf = inject_race(b, 'black')
        self.assertTrue(cf['dialogue'].startswith('Doctor: Before we start'))
        self.assertEqual([f['value'] for f in race_evidence(cf['dialogue'])], ['black'])
        self.assertEqual(cf['counterfactual']['gold'], {'race': 'black'})
        self.assertEqual(cf['case_id'], 'cf:race_injection:val:1')
        self.assert_replayable(b, cf)

    def test_inject_skips_dialogue_with_race(self):
        self.assertIsNone(inject_race(base("Patient: I'm white. I have a cough."), 'black'))

    def test_remove_race(self):
        b = base("Doctor: How old are you? Patient: I am forty year old Asian. I have a cough.")
        cf = remove_race(b)
        self.assertFalse(has_race_vocabulary(cf['dialogue']))
        self.assertIn('forty year old', cf['dialogue'])
        self.assertEqual(cf['counterfactual']['gold'], {'race': None})
        self.assert_replayable(b, cf)

    def test_shift_age_words_curly(self):
        b = base('Doctor: How old are you? Patient: I’m fifty seven years old.')
        cf = shift_age(b)
        self.assertIn('sixty four', cf['dialogue'])
        self.assertEqual(cf['counterfactual']['gold'], {'age': 64, 'original_age': 57})
        self.assert_replayable(b, cf)

    def test_shift_age_digits(self):
        self.assertIn('64', shift_age(base('Patient: I am 57 years old.'))['dialogue'])

    def test_shift_age_two_values_ineligible(self):
        self.assertIsNone(shift_age(base('Patient: I am 57 years old. Patient: I am 58 years old.')))

    def test_swap_pronouns(self):
        b = base('Doctor: Hi sir. The patient says he hurt his knee.')
        cf = swap_pronouns(b)
        self.assertIn("ma'am", cf['dialogue'])
        self.assertIn('she hurt her knee', cf['dialogue'])
        self.assertEqual(cf['counterfactual']['gold'], {'pronoun': 'feminine'})
        self.assert_replayable(b, cf)

    def test_swap_skips_feminine(self):
        self.assertIsNone(swap_pronouns(base('Doctor: The patient says she hurt her knee.')))


class Packet(unittest.TestCase):
    def cases(self):
        return [base(f'Doctor: How old are you? Patient: I am {40 + i} years old. I have a cough.', f'val:{i}')
                for i in range(16)]

    def test_deterministic_and_rotated(self):
        a = build_packet(self.cases(), words(0), seed=20261109)
        b = build_packet(self.cases(), words(0), seed=20261109)
        self.assertEqual([c['case_id'] for c in a], [c['case_id'] for c in b])
        groups = Counter(c['counterfactual']['gold']['race'] for c in a
                         if c['counterfactual']['family'] == 'race_injection')
        self.assertEqual(set(groups), set(GROUPS))
        self.assertLessEqual(max(groups.values()) - min(groups.values()), 1)

    def test_edit_beyond_window_excluded(self):
        long = base('Patient: ' + 'cough ' * 560 + 'Doctor: How old are you? Patient: I am 57 years old.')
        packet = build_packet([long], words(0), seed=20261109)
        self.assertEqual([c['counterfactual']['family'] for c in packet], ['race_injection'])

    def test_cap(self):
        packet = build_packet(self.cases(), words(0), seed=20261109, cap=20)
        self.assertEqual(len(packet), 20)
        self.assertEqual(sum(c['counterfactual']['family'] == 'race_injection' for c in packet), 16)


class Scoring(unittest.TestCase):
    def cf(self, family, gold, prediction):
        return {'case_id': f'cf:{family}:val:1', 'dialogue': 'Patient: hi.', 'prediction': prediction,
                'counterfactual': {'family': family, 'gold': gold}}

    def test_race_outcomes(self):
        s = score_counterfactual([
            self.cf('race_injection', {'race': 'black'}, 'This is a 40-year-old white male.'),
            self.cf('race_injection', {'race': 'black'}, 'This is a 40-year-old male.'),
            self.cf('race_injection', {'race': 'black'}, 'This is a 40-year-old Black male.'),
            self.cf('race_removal', {'race': None}, 'This is a 40-year-old Asian male.'),
        ])
        inj = s['families']['race_injection']
        self.assertEqual((inj['contradicted'], inj['omitted'], inj['faithful'], inj['n']), (1, 1, 1, 3))
        self.assertEqual(s['families']['race_removal']['inserted'], 1)
        self.assertTrue(s['race_by_group']['underpowered'])

    def test_age_reverted(self):
        s = score_counterfactual([self.cf('age_shift', {'age': 64, 'original_age': 57}, 'The patient is 57 years old.')])
        self.assertEqual(s['families']['age_shift']['contradicted'], 1)
        self.assertEqual(s['families']['age_shift']['reverted_to_original'], 1)


if __name__ == '__main__':
    unittest.main()
