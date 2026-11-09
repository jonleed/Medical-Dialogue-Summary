"""v3 matching, dialogue-cluster bootstrap metrics, summarizer rates, agreement and reporting rules."""
import unittest

from factuality.v3_scoring import (ERROR_TYPES, agreement, detector_metrics, headline_caveat, headline_wording,
                                   match_claims, match_omissions, summarizer_rates)


def span(s, e):
    return {'start': s, 'end': e, 'text': 'x' * (e - s)}


def human(label, ft='age', s=0, e=5, subject='patient'):
    return {'label': label, 'fact_type': ft, 'subject': subject, 'generated_span': span(s, e)}


def det(label, ft='age', s=0, e=5, subject='patient', issue=None):
    return {'label': label, 'fact_type': ft, 'subject': subject, 'generated_span': span(s, e), 'issue': issue}


def rows_for(error_type, rows):
    return [r for r in rows if r['error_type'] == error_type]


class Matching(unittest.TestCase):
    def test_true_positive(self):
        [r] = rows_for('age_contradiction', match_claims([human('contradicted')], [det('contradicted')]))
        self.assertEqual((r['human_positive'], r['detector_positive'], r['label_correct']), (True, True, True))

    def test_abstention_is_a_miss(self):
        [r] = rows_for('age_contradiction', match_claims([human('contradicted')], [det('unclear')]))
        self.assertEqual((r['human_positive'], r['detector_positive']), (True, None))

    def test_unmatched_human_claim_is_a_miss(self):
        [r] = rows_for('age_contradiction', match_claims([human('contradicted')], [det('contradicted', s=20, e=25)])
                       )[:1]
        self.assertEqual((r['human_positive'], r['detector_positive'], r['extra']), (True, None, False))

    def test_unmatched_detector_claim_is_extra(self):
        rows = rows_for('age_contradiction', match_claims([], [det('contradicted')]))
        self.assertTrue(rows[0]['extra'])

    def test_type_convention_mismatch_is_not_an_error(self):
        rows = match_claims([human('contradicted', ft='age', subject='relative')],
                            [det('contradicted', ft='quantity', subject='relative')])
        [r] = rows_for('age_contradiction', rows)
        self.assertEqual((r['detector_positive'], r['label_correct'], r['type_convention_match']), (True, True, False))

    def test_fallback_match_of_other_type_does_not_count_as_hit(self):
        rows = match_claims([human('contradicted', ft='age')], [det('contradicted', ft='sex')])
        [r] = rows_for('age_contradiction', rows)
        self.assertNotEqual(r['detector_positive'], True)

    def test_race_inserted(self):
        [r] = rows_for('race_inserted', match_claims([human('unsupported', ft='race_ethnicity')],
                                                     [det('unsupported', ft='race_ethnicity', issue='race_inserted')]))
        self.assertEqual((r['human_positive'], r['detector_positive']), (True, True))

    def test_cross_dimension_flag_counts_for_sex_reference_error(self):
        [r] = rows_for('cross_dimension_inconsistency', match_claims(
            [human('contradicted', ft='sex')], [det('unclear', ft='sex', issue='cross_dimension_inconsistency')]))
        self.assertEqual(r['detector_positive'], True)

    def test_all_error_types_emitted(self):
        types = {r['error_type'] for r in match_claims([human('supported')], [det('supported')])}
        self.assertEqual(types, set(ERROR_TYPES) - {'omission'})


class Omissions(unittest.TestCase):
    def test_match_by_source_span(self):
        facts = [{'label': 'omitted', 'source_spans': [span(10, 20)]}, {'label': 'present', 'source_spans': [span(30, 40)]},
                 {'label': 'excluded', 'source_spans': [span(50, 60)]}]
        inventory = [{'status': 'omitted', 'source_span': span(12, 18)}, {'status': 'omitted', 'source_span': span(31, 35)}]
        rows = match_omissions(facts, inventory)
        self.assertEqual([(r['human_positive'], r['detector_positive']) for r in rows], [(True, True), (False, True)])

    def test_unlisted_fact_is_miss_not_false_positive(self):
        [r] = match_omissions([{'label': 'omitted', 'source_spans': [span(0, 5)]}], [])
        self.assertIsNone(r['detector_positive'])


class Metrics(unittest.TestCase):
    def rows(self):
        out = {}
        for i in range(10):
            rows = match_claims([human('contradicted')], [det('contradicted' if i < 7 else 'unclear')])
            rows += match_claims([human('supported', s=10, e=15)], [det('contradicted' if i == 0 else 'supported', s=10, e=15)])
            out[f'd{i}'] = rows
        return out

    def test_point_estimates(self):
        m = detector_metrics(self.rows(), resamples=200)['age_contradiction']
        self.assertEqual(m['positives'], 10)
        self.assertAlmostEqual(m['recall'], 0.7)
        self.assertAlmostEqual(m['precision'], 7 / 8)
        self.assertAlmostEqual(m['abstention_rate'], 0.3)

    def test_bootstrap_deterministic_and_bounded(self):
        a = detector_metrics(self.rows(), resamples=200)['age_contradiction']['ci95']
        b = detector_metrics(self.rows(), resamples=200)['age_contradiction']['ci95']
        self.assertEqual(a, b)
        self.assertLessEqual(a['recall'][0], 0.7)
        self.assertGreaterEqual(a['recall'][1], 0.7)

    def test_no_positives(self):
        m = detector_metrics({'d0': match_claims([human('supported')], [det('supported')])}, resamples=10)
        self.assertIsNone(m['age_contradiction']['recall'])


class Rates(unittest.TestCase):
    def ann(self, cid, run, labels, facts=()):
        return {'case_id': cid, 'run': run, 'claims': [{'label': l} for l in labels],
                'expected_facts': [{'label': f} for f in facts]}

    def test_random_only(self):
        anns = [self.ann('val:1', 'a', ['contradicted', 'supported'], ['omitted', 'present']),
                self.ann('val:2', 'a', ['unsupported', 'unclear']),
                self.ann('val:3', 'a', ['contradicted', 'contradicted'])]
        r = summarizer_rates(anns, random_ids={'val:1', 'val:2'}, resamples=50)['a']
        self.assertAlmostEqual(r['contradiction']['rate'], 1 / 3)
        self.assertAlmostEqual(r['unsupported']['rate'], 1 / 3)
        self.assertAlmostEqual(r['omission']['rate'], 1 / 2)
        self.assertEqual(r['dialogues'], 2)


class Agreement(unittest.TestCase):
    def test_kappa_needs_five_per_cell(self):
        self.assertIsNone(agreement([True, False, True], [True, False, False])['kappa'])

    def test_kappa(self):
        a = [True] * 10 + [False] * 10
        b = [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5
        r = agreement(a, b)
        self.assertEqual(r['raw'], 0.5)
        self.assertAlmostEqual(r['kappa'], 0.0)


class Wording(unittest.TestCase):
    def test_detects(self):
        m = {'positives': 12, 'precision': 0.9, 'recall': 0.85, 'ci95': {'precision': [0.7, 1.0], 'recall': [0.6, 1.0]}}
        self.assertTrue(headline_wording('age_contradiction', m).startswith('detects age contradiction'))

    def test_screens(self):
        m = {'positives': 12, 'precision': 0.9, 'recall': 0.7, 'ci95': {'precision': [0.7, 1.0], 'recall': [0.5, 0.9]}}
        self.assertEqual(headline_wording('age_contradiction', m),
                         'screens for age contradiction, measured recall 0.70 (95% CI 0.50–0.90)')

    def test_insufficient(self):
        self.assertEqual(headline_wording('race_contradiction', {'positives': 3}),
                         'insufficient positives to estimate detector recall')

    def test_caveat_uses_actual_heldout_count(self):
        self.assertEqual(headline_caveat(29), 'single non-clinician annotator, 29 held-out dialogues')


if __name__ == '__main__':
    unittest.main()


class Assembly(unittest.TestCase):
    def test_rows_grouped_by_dialogue_across_runs(self):
        from factuality.v3_scoring import rows_by_dialogue
        anns = [{'case_id': 'val:1', 'run': 'a', 'claims': [human('contradicted')], 'expected_facts': []},
                {'case_id': 'val:1', 'run': 'b', 'claims': [human('supported')], 'expected_facts': []}]
        detectors = {('a', 'val:1'): {'claims': [{'proposed_label': 'contradicted', 'fact_type': 'age',
                                                  'generated_span': span(0, 5), 'issue': 'demographic_comparison'}]},
                     ('b', 'val:1'): {'claims': [det('supported')]}}
        rows = rows_by_dialogue(anns, detectors)
        self.assertEqual(list(rows), ['val:1'])
        age = rows_for('age_contradiction', rows['val:1'])
        self.assertEqual([(r['human_positive'], r['detector_positive']) for r in age], [(True, True), (False, False)])

    def test_missing_detector_output_counts_as_misses(self):
        from factuality.v3_scoring import rows_by_dialogue
        anns = [{'case_id': 'val:1', 'run': 'a', 'claims': [human('contradicted')],
                 'expected_facts': [{'label': 'omitted', 'source_spans': [span(0, 5)]}]}]
        rows = rows_by_dialogue(anns, {})['val:1']
        self.assertTrue(all(r['detector_positive'] is None for r in rows))
        self.assertIn('omission', {r['error_type'] for r in rows})

    def test_judge_on_counterfactual(self):
        from factuality.v3_scoring import judge_on_counterfactual
        cf = {'case_id': 'cf:race_injection:val:1', 'dialogue': 'Patient: hi.',
              'prediction': 'This is a 40-year-old white male.',
              'counterfactual': {'family': 'race_injection', 'gold': {'race': 'black'}}}
        white = cf['prediction'].index('white')
        judged = {'claims': [det('contradicted', ft='race_ethnicity', s=white, e=white + 5)]}
        r = judge_on_counterfactual([('cf-a', cf)], {('cf-a', cf['case_id']): judged})
        self.assertEqual(r['race_injection'], {'n': 1, 'correct': 1, 'missed': 0, 'wrong_label': 0})


class Assemble(unittest.TestCase):
    def test_public_metrics_shape_and_no_text(self):
        import json
        from factuality.v3_scoring import assemble_metrics
        anns = [{'case_id': f'val:{i}', 'run': 'a', 'claims': [dict(human('contradicted' if i % 2 else 'supported'), text='secret claim')],
                 'expected_facts': [{'label': 'omitted', 'source_spans': [span(0, 5)], 'text': 'secret fact'}]}
                for i in range(6)]
        sample = {'random': [f'val:{i}' for i in range(6)], 'enrichment': [], 'calibration': ['val:0', 'val:1'],
                  'heldout': [f'val:{i}' for i in range(2, 6)]}
        detectors = {k: {('a', f'val:{i}'): {'claims': [det('contradicted')], 'section_inventory': []} for i in range(6)}
                     for k in ('rules', 'claude')}
        trials = {'t1': detectors['claude'], 't2': detectors['claude']}
        m = assemble_metrics(anns, sample, detectors, trials=trials, secondary=detectors['claude'], resamples=20)
        self.assertEqual(m['heldout']['dialogues'], 4)
        self.assertEqual(m['heldout']['claude']['age_contradiction']['positives'], 2)
        self.assertNotIn('omission', m['heldout']['rules'])
        self.assertIn('omission', m['heldout']['claude'])
        self.assertEqual(m['agreement']['claude_trials']['raw'], 1.0)
        self.assertIn('headline', m['heldout'])
        self.assertTrue(m['heldout']['caveat'].endswith('4 held-out dialogues'))
        self.assertNotIn('secret', json.dumps(m))
