import unittest

class SemanticExecutionTests(unittest.TestCase):
    def test_exact_quotes_are_anchored_without_fuzzy_evidence(self):
        from factuality.semantic_run import anchor
        self.assertEqual(anchor('abc xyz abc','abc',1),{'start':8,'end':11,'text':'abc'})
        for quote,occurrence in [('abcd',0),('abc',2),('',0),('abc',True)]:
            with self.assertRaises(ValueError):anchor('abc xyz abc',quote,occurrence)

    def test_inventory_normalization_validates_both_texts(self):
        from factuality.semantic_run import normalize_inventory
        task={'task_id':'x','task_hash':'h','source':'Patient: I am 57.','prediction':'67 years old.'}
        raw={'claims_complete':True,'claims':[{'text':'Age67','fact_type':'age','generated_quote':'67','occurrence':0,'label':'contradicted','rationale':'57 differs from67','evidence_quotes':['I am 57.']}],'expected_facts':[]}
        out=normalize_inventory(task,raw)
        self.assertEqual(out['claims'][0]['generated_span']['text'],'67')
        raw['claims'][0]['evidence_quotes']=['I am 67.']
        with self.assertRaises(ValueError):normalize_inventory(task,raw)

    def test_omission_reference_is_not_inferred_from_blank_annotation(self):
        from factuality.semantic_run import omission_reference
        self.assertTrue(omission_reference('Missing claim: smoking history.'))
        self.assertTrue(omission_reference('The note is omitting the reassurance.'))
        self.assertIsNone(omission_reference(''))
        self.assertIsNone(omission_reference('Fabricated claim - not mentioned in reference.'))

if __name__=='__main__':unittest.main()

class ConservativeControls(unittest.TestCase):
    def test_negated_or_uncertain_omission_mentions_are_not_gold_positives(self):
        from factuality.semantic_run import omission_reference
        for text in ['No omissions.', 'No missing context was identified.', 'The note is complete without omissions.', 'Possible missing claim: uncertain.', 'Missing claim: none.']:
            self.assertIsNone(omission_reference(text),text)

    def test_repeated_evidence_is_not_assigned_to_first_speaker(self):
        from factuality.semantic_run import normalize_inventory
        task={'source':'Doctor: I have pain.\nPatient: I have pain.','prediction':'Patient has pain.'}
        raw={'claims_complete':True,'claims':[{'text':'Patient has pain','fact_type':'symptom','generated_quote':'pain','occurrence':0,'label':'supported','rationale':'Patient reports pain','evidence_quotes':['I have pain.']}],'expected_facts':[]}
        claim=normalize_inventory(task,raw)['claims'][0]
        self.assertEqual(len(claim['source_spans']),2)
        self.assertTrue(claim['ambiguous_source_evidence'])
        self.assertEqual(claim['label'],'unclear')
        self.assertEqual(claim['proposed_label'],'supported')

class ResumabilityControls(unittest.TestCase):
    def test_failed_calls_are_archived_and_can_resume(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from types import SimpleNamespace
        from factuality.semantic_run import run_one
        job={'id':'sample','kind':'plan','task':{'source':'Patient: fever.','requested_section':'chief complaint'}}
        with tempfile.TemporaryDirectory() as d:
            failure=SimpleNamespace(stdout='',stderr='transport failed',returncode=1)
            with patch('factuality.semantic_run.subprocess.run',return_value=failure):
                self.assertEqual(run_one(job,d)['status'],'failed')
            def success(cmd,**kwargs):
                Path(cmd[cmd.index('--output-last-message')+1]).write_text(json.dumps({'facts':[]}))
                return SimpleNamespace(stdout='{"type":"turn.completed","usage":{}}\n',stderr='',returncode=0)
            with patch('factuality.semantic_run.subprocess.run',side_effect=success):
                self.assertEqual(run_one(job,d)['status'],'complete')
            self.assertTrue((Path(d)/'sample'/'retry-archive-000'/'result.json').exists())

    def test_risk_metrics_distinguish_low_risk_from_review_required(self):
        from factuality.semantic_study import risk_metrics
        m=risk_metrics([(1,1),(2,3),(3,2),(4,4)])
        self.assertEqual(m['accuracy'],.5)
        self.assertEqual(m['review_required'],{'tp':1,'fn':1,'fp':1,'tn':1,'sensitivity':.5,'specificity':.5,'false_positive_rate':.5})

class BatchIsolationControls(unittest.TestCase):
    def test_batch_inventory_cannot_borrow_another_patients_evidence(self):
        from factuality.semantic_run import validate
        task={'cases':[{'id':'0','data':{'source':'Patient is 57.','prediction':'67 years old.','plans':[]}},{'id':'1','data':{'source':'Patient is 67.','prediction':'67 years old.','plans':[]}}]}
        claim={'text':'age67','fact_type':'age','generated_quote':'67','occurrence':0,'label':'supported','rationale':'matches','evidence_quotes':['Patient is 67.']}
        raw={'items':[{'id':'0','result':{'claims_complete':True,'claims':[claim],'expected_facts':[]}},{'id':'1','result':{'claims_complete':True,'claims':[claim],'expected_facts':[]}}]}
        with self.assertRaises(ValueError):validate('batch_inventory',task,raw)

    def test_batch_requires_every_case_exactly_once(self):
        from factuality.semantic_run import validate
        task={'cases':[{'id':'0','data':{'source':'source','prediction':'note','plans':[]}}]}
        with self.assertRaises(ValueError):validate('batch_inventory',task,{'items':[]})

class ExtractionDenominatorControls(unittest.TestCase):
    def test_correct_label_for_wrong_fact_type_is_an_extraction_miss(self):
        from factuality.semantic_results import match_target
        c={'fact_type':'age','label':'contradicted','generated_span':{'start':0,'end':20}}
        self.assertEqual(match_target([c],{'start':3,'end':5},'quantity'),('not_extracted',False))

    def test_conflicting_same_type_matches_are_not_selected_by_gold(self):
        from factuality.semantic_results import match_target
        c={'fact_type':'age','label':'supported','generated_span':{'start':0,'end':20}}
        self.assertEqual(match_target([c,dict(c,label='contradicted')],{'start':3,'end':5},'age'),('not_graded',True))
