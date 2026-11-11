"""Seeded annotation sample: 40 random + 10 enrichment, 15 calibration / 35 held-out by dialogue."""
import unittest

from factuality.sampling import SEEN, enrichment_eligible, select_sample

IDS = [f'val:{i}' for i in range(100)]
ELIGIBLE = {f'val:{i}' for i in range(0, 100, 3)}


class Sample(unittest.TestCase):
    def setUp(self):
        self.s = select_sample(IDS, ELIGIBLE)

    def test_deterministic(self):
        self.assertEqual(self.s, select_sample(IDS, ELIGIBLE))

    def test_sizes(self):
        self.assertEqual(len(self.s['random']), 40)
        self.assertEqual(len(self.s['enrichment']), 10)
        self.assertEqual(len(self.s['calibration']), 15)
        self.assertEqual(len(self.s['heldout']), 35)

    def test_enrichment_from_eligible_and_disjoint(self):
        self.assertTrue(set(self.s['enrichment']) <= ELIGIBLE)
        self.assertFalse(set(self.s['enrichment']) & set(self.s['random']))

    def test_splits_partition_sample(self):
        cal, held = set(self.s['calibration']), set(self.s['heldout'])
        self.assertFalse(cal & held)
        self.assertEqual(cal | held, set(self.s['random']) | set(self.s['enrichment']))

    def test_seen_never_heldout(self):
        self.assertFalse(SEEN & set(self.s['heldout']))
        self.assertTrue(SEEN & (set(self.s['random']) | set(self.s['enrichment'])) <= set(self.s['calibration']))

    def test_inclusion_probabilities(self):
        p = self.s['inclusion_probability']
        self.assertEqual(p[self.s['random'][0]], 0.4)
        self.assertEqual(p[self.s['enrichment'][0]], min(1.0, 10 / len(ELIGIBLE - set(self.s['random']))))

    def test_too_many_seen_raises(self):
        with self.assertRaises(ValueError):
            select_sample(IDS, ELIGIBLE, seen=frozenset(IDS[:60]))

    def test_fewer_eligible_takes_all(self):
        s = select_sample(IDS, {'val:1', 'val:2'} - set(select_sample(IDS, set())['random']))
        self.assertLessEqual(len(s['enrichment']), 2)


class Eligibility(unittest.TestCase):
    def test_rules(self):
        rows = [
            {'run': 'a', 'case_id': 'val:1', 'suggestions': [{'fact_type': 'age', 'proposed_label': 'contradicted', 'issue': 'x'}]},
            {'run': 'a', 'case_id': 'val:2', 'suggestions': [{'fact_type': 'race_ethnicity', 'proposed_label': 'supported', 'issue': 'x'}]},
            {'run': 'b', 'case_id': 'val:3', 'suggestions': [{'fact_type': 'sex', 'proposed_label': 'unclear', 'issue': 'cross_dimension_inconsistency'}]},
            {'run': 'a', 'case_id': 'val:4', 'suggestions': [{'fact_type': 'pronoun', 'proposed_label': 'unclear', 'issue': 'pronoun_not_established'}]},
            {'run': 'c', 'case_id': 'val:5', 'suggestions': [{'fact_type': 'age', 'proposed_label': 'contradicted', 'issue': 'x'}]},
        ]
        self.assertEqual(enrichment_eligible(rows, runs={'a', 'b'}), {'val:1', 'val:2', 'val:3'})



class TopUp(unittest.TestCase):
    def test_tops_heldout_up_to_target_without_changing_original_draw(self):
        base = select_sample(IDS, ELIGIBLE, seen=frozenset())
        s = select_sample(IDS, ELIGIBLE, seen=frozenset(), heldout_target=40)
        self.assertEqual((s['random'], s['enrichment'], s['calibration']), (base['random'], base['enrichment'], base['calibration']))
        self.assertEqual(len(s['heldout']), 40)
        self.assertEqual(len(s['top_up']), 5)
        self.assertTrue(set(s['top_up']) <= set(s['heldout']))
        self.assertFalse(set(s['top_up']) & (set(base['random']) | set(base['enrichment'])))
        self.assertEqual(s, select_sample(IDS, ELIGIBLE, seen=frozenset(), heldout_target=40))

    def test_top_up_never_draws_seen(self):
        s = select_sample(IDS, set(), seen=SEEN, heldout_target=40)
        self.assertFalse(set(s['top_up']) & SEEN)
        pool = set(IDS) - set(s['random']) - set(s['enrichment']) - SEEN
        self.assertEqual(len(s['top_up']), 15)
        self.assertEqual(s['inclusion_probability'][s['top_up'][0]], 15 / len(pool))

    def test_no_top_up_by_default(self):
        self.assertEqual(select_sample(IDS, ELIGIBLE)['top_up'], [])


if __name__ == '__main__':
    unittest.main()
