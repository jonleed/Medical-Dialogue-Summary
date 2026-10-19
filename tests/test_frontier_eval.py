"""Evaluator quality, immutable challenge, and evidence-judge protocol invariants."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class MetaEvaluationTests(unittest.TestCase):
    def test_missing_predictions_stay_in_recall_denominator(self):
        from factuality.meta import evaluate
        rows=[{'id':'a','gold':'contradicted','pred':'contradicted','group':'g1'},
              {'id':'b','gold':'contradicted','pred':'not_extracted','group':'g2'},
              {'id':'c','gold':'supported','pred':'unclear','group':'g3'}]
        r=evaluate(rows,bootstrap=50)
        self.assertEqual(r['contradiction_sensitivity']['denominator'],2)
        self.assertEqual(r['contradiction_sensitivity']['rate'],.5)
        self.assertEqual(r['extraction_coverage']['rate'],2/3)
        self.assertEqual(r['accuracy']['rate'],1/3)
        self.assertEqual(r['false_contradiction_rate_on_supported']['rate'],0)

    def test_unassessed_visible_evidence_is_not_missing_claim_extraction(self):
        from factuality.meta import evaluate
        r=evaluate([{'id':'a','gold':'unsupported','pred':'not_assessed','extracted':True,'group':'f'}],bootstrap=0)
        self.assertEqual(r['extraction_coverage']['rate'],1)
        self.assertEqual(r['label_coverage']['rate'],0)
        self.assertEqual(r['accuracy']['rate'],0)

    def test_no_labels_is_na_not_a_perfect_score(self):
        from factuality.meta import evaluate
        r=evaluate([],bootstrap=20)
        self.assertIsNone(r['accuracy']['rate'])
        self.assertIsNone(r['macro_f1'])
        self.assertIsNone(r['cluster_bootstrap_95']['accuracy'])

    def test_matched_pairs_and_cluster_bootstrap(self):
        from factuality.meta import evaluate
        rows=[{'id':'a','gold':'supported','pred':'supported','group':'family','pair_id':'p','variant':'correct'},
              {'id':'b','gold':'contradicted','pred':'unclear','group':'family','pair_id':'p','variant':'corrupted'}]
        r=evaluate(rows,bootstrap=30)
        self.assertEqual(r['matched_pairs_both_correct']['denominator'],1)
        self.assertEqual(r['matched_pairs_both_correct']['numerator'],0)
        self.assertEqual(r['cluster_bootstrap_95']['groups'],1)
        self.assertEqual(r['cluster_bootstrap_95'],evaluate(rows,bootstrap=30)['cluster_bootstrap_95'])

    def test_reject_duplicate_items_or_unknown_labels(self):
        from factuality.meta import evaluate
        row={'id':'a','gold':'supported','pred':'supported','group':'f'}
        for rows in [[row,row],[dict(row,gold='wrong')],[dict(row,pred='pass')]]:
            with self.assertRaises(ValueError):evaluate(rows)


class ChallengeTests(unittest.TestCase):
    def test_freeze_then_score_preserves_checker_and_gold(self):
        from factuality.challenge import freeze, run
        with tempfile.TemporaryDirectory() as d:
            frozen=Path(d)/'frozen';out=Path(d)/'score'
            manifest=freeze(frozen)
            self.assertGreater(manifest['cases'],100)
            report=run(frozen,out)
            self.assertEqual(report['checker_hashes'],manifest['checker_hashes'])
            self.assertGreater(report['full_source']['all']['contradiction_sensitivity']['denominator'],50)
            self.assertEqual(report['evaluator_status'],'unvalidated_screening_baseline')
            with self.assertRaises(ValueError):freeze(frozen)
            with self.assertRaises(ValueError):run(frozen,out)

    def test_quality_gate_fails_while_preserving_capability_results(self):
        import subprocess,sys
        from factuality.challenge import freeze
        with tempfile.TemporaryDirectory() as d:
            frozen=Path(d)/'frozen';out=Path(d)/'results';freeze(frozen)
            run=subprocess.run([sys.executable,'-m','factuality','challenge-run','--frozen',str(frozen),'--out',str(out),'--require-targets'],capture_output=True,text=True)
            self.assertEqual(run.returncode,2)
            self.assertTrue((out/'metrics.json').exists())

    def test_tampered_frozen_gold_is_rejected(self):
        from factuality.challenge import freeze, run
        with tempfile.TemporaryDirectory() as d:
            frozen=Path(d)/'frozen';freeze(frozen)
            p=frozen/'cases.jsonl';p.write_text(p.read_text().replace('contradicted','supported',1))
            with self.assertRaises(ValueError):run(frozen,Path(d)/'out')

    def test_pairs_are_single_controlled_edits_and_no_real_cases(self):
        from factuality.challenge import build
        from collections import defaultdict
        pairs=defaultdict(list)
        for row in build():
            self.assertEqual(row['case']['dataset_kind'],'synthetic')
            if row['pair_id']:pairs[row['pair_id']].append(row)
        for pair in pairs.values():
            self.assertEqual(len(pair),2)
            good,bad=pair
            self.assertEqual(good['case']['dialogue'],bad['case']['dialogue'])
            edit=bad['edit'];text=good['case']['prediction']
            self.assertEqual(text[edit['start']:edit['end']],edit['before'])
            self.assertEqual(text[:edit['start']]+edit['after']+text[edit['end']:],bad['case']['prediction'])

    def test_no_duplicate_evidence_claim_tasks_and_atomic_demographic_targets(self):
        from factuality.challenge import build
        rows=build()
        keys=[(r['case']['dialogue'],r['case']['prediction'],r['case']['model_input']['text'],r['target_type']) for r in rows]
        self.assertEqual(len(keys),len(set(keys)))
        for row in rows:
            if row['target_type']=='pronoun':self.assertIn(row['target_span']['text'].lower(),{'he','she'})
            if row['family']=='medication_identity':self.assertIn('do not take metformin',row['case']['dialogue'])
            if row['family']=='diagnosis_identity':self.assertIn('does not have asthma',row['case']['dialogue'])
            if row['family'] in {'gender_explicit_self','gender_nonbinary_alias'}:self.assertIn('do not identify as a man',row['case']['dialogue'])
            if row['family'].startswith('sex_explicit'):self.assertIn('not male',row['case']['dialogue'])


class JudgeProtocolTests(unittest.TestCase):
    def fixture(self):
        from factuality.judging import claim_task
        return claim_task('private-biobart-id','Patient: I am 42 years old.',
                          'The patient is 24 years old.','age','full_source')

    def response(self,task,label='contradicted'):
        return {'task_id':task['task_id'],'task_hash':task['task_hash'],
          'trial':0,'reviewer':{'kind':'model','id':'judge-A','model_revision':'exact-revision','model_family':'family-A','parameters':{'temperature':0}},
          'label':label,'rationale':'Source states 42, whereas claim says 24.',
          'evidence':[{'start':9,'end':27,'text':'I am 42 years old.'}]}

    def test_blind_task_has_no_gold_or_model_identity(self):
        task=self.fixture();encoded=json.dumps(task)
        self.assertNotIn('biobart',encoded)
        self.assertNotIn('gold',task)
        self.assertIn('Treat',task['rubric'])

    def test_validate_exact_evidence_and_judge_identity(self):
        from factuality.judging import validate_response
        task=self.fixture();response=self.response(task)
        validate_response(task,response)
        for mutate in [lambda x:x.update(task_hash='stale'),lambda x:x.update(evidence=[]),
          lambda x:x['evidence'][0].update(text='invented'),lambda x:x['reviewer'].update(model_revision=''),
          lambda x:x.update(label='correct'),lambda x:x.update(trial=True)]:
            bad=copy.deepcopy(response);mutate(bad)
            with self.assertRaises(ValueError):validate_response(task,bad)

    def test_duplicate_same_judge_cannot_become_consensus(self):
        from factuality.judging import judge_report
        task=self.fixture();r=self.response(task)
        with self.assertRaises(ValueError):judge_report([task],[r,r])
        report=judge_report([task],[r])
        self.assertEqual(report['jury']['consensus_tasks'],0)
        self.assertIsNone(report['human_calibrated_accuracy'])

    def test_disagreement_is_queued_not_resolved_by_majority(self):
        from factuality.judging import judge_report
        task=self.fixture();a=self.response(task);b=self.response(task,'supported');b['reviewer']['id']='judge-B';b['reviewer']['model_revision']='other-model';b['reviewer']['model_family']='family-B'
        r=judge_report([task],[a,b])
        self.assertEqual(r['jury']['disagreement_tasks'],1)
        self.assertEqual(r['jury']['consensus_tasks'],0)

    def test_pending_judgments_are_not_passes(self):
        from factuality.judging import judge_report
        task=self.fixture();r=judge_report([task],[])
        self.assertEqual(r['tasks'],1)
        self.assertEqual(r['responses'],0)
        self.assertEqual(r['jury']['consensus_tasks'],0)
        self.assertIsNone(r['human_calibrated_accuracy'])

    def test_judge_trial_parameters_are_not_overwritten(self):
        from factuality.judging import judge_report,claim_task
        task=self.fixture();a=self.response(task);b=copy.deepcopy(a);b['trial']=1;b['reviewer']['parameters']={'temperature':1}
        report=judge_report([task],[a,b])
        self.assertEqual([j['reviewer']['parameters']['temperature'] for j in report['judges']],[0,1])
        other=claim_task('other','Patient: I am 42 years old.','The patient is 24 years old.','age','full_source')
        b['task_id']=other['task_id'];b['task_hash']=other['task_hash'];b['trial']=0
        with self.assertRaises(ValueError):judge_report([task,other],[a,b])

    def test_missing_generation_cannot_complete_as_empty_inventory(self):
        from factuality.judging import prepare
        from factuality.io import write_jsonl
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cases.jsonl';write_jsonl(p,[{'case_id':'missing','dialogue':'Patient: Hello.','prediction':None}])
            with self.assertRaises(ValueError):prepare([p],Path(d)/'judges')

    def test_packets_separate_source_views_and_inventory_from_known_claims(self):
        from factuality.judging import prepare
        from factuality.io import write_jsonl,read_jsonl
        c={'case_id':'x','dialogue':'Patient: Hello.','prediction':'No fever.','requested_section':'HPI',
           'model_input':{'text':'Patient: Hello.','truncated':False}}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cases.jsonl';write_jsonl(p,[c]);out=Path(d)/'judges';prepare([p],out)
            packet_files=list((out/'reviewer-A').rglob('tasks.jsonl'))
            self.assertTrue(packet_files)
            for path in packet_files:
                rows=read_jsonl(path)
                self.assertLessEqual(len({r['view'] for r in rows}),1)
                self.assertLessEqual(len({r['kind'] for r in rows}),1)
            self.assertFalse((out/'reviewer-A'/'tasks.jsonl').exists())


if __name__=='__main__':unittest.main()
