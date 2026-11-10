"""Offline pipeline invariants, synthetic frames and real sampling parity (no weights)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from factuality.export import sample_cases
from factuality.io import case_hash, read_jsonl
from factuality.review import blank_annotation, blind_plans, write_queue
from factuality.scoring import score, validate_annotation
from test_factuality import case, annotation


class PipelineDiagnostics(unittest.TestCase):
    def test_group_sort_sample_and_exam_mapping(self):
        import pandas as pd
        frame = pd.DataFrame([{'ID':i,'section_header':'EXAM', 'section_text':'Normal.',
                               'dialogue':f'Patient: I am {20+i} years old.'} for i in range(12)])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'val.csv'; frame.to_csv(p,index=False)
            cases = sample_cases(p,n=5)
            expected = frame.assign(dialogue_id=frame.ID.astype(str)).groupby('dialogue_id').first().reset_index().sample(n=5, random_state=42)
            self.assertEqual([c['source']['dialogue_id'] for c in cases],expected.dialogue_id.tolist())
            self.assertEqual(cases[0]['requested_section'],'physical exam')
            self.assertEqual(cases[0]['notebook_section'],'exam')
            self.assertEqual(sample_cases(p,n=5),cases)

    def test_blind_queue_and_pending_round_trip(self):
        c = case(); c['dataset_kind']='real'; c['requested_section']='HPI'; c['reference']='age'
        p = blind_plans([c])[0]
        self.assertNotIn('prediction',p)
        self.assertNotIn('model_input',p)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'queue'; summary = write_queue([c],out)
            a = read_jsonl(out/'annotations.jsonl')
            metrics = score([c],a)
            self.assertEqual(metrics['human']['pending_cases'],1)
            self.assertEqual(summary['human_reviewed_cases'],0)
            self.assertIn('pending',(out/'review.html').read_text())
            with self.assertRaises(ValueError): write_queue([c],out)

    def test_html_escaping(self):
        c = case(prediction='<script>bad()</script>'); c.update(requested_section='HPI',reference='')
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'queue'; write_queue([c],out)
            self.assertNotIn('<script>',(out/'review.html').read_text())

    def test_unresolved_complete_not_passing(self):
        c = case(); a = annotation(c); a['unresolved']=['Attribution uncertain']
        self.assertEqual(score([c],[a])['synthetic']['passing_complete_cases'],0)

    def test_no_claims_for_missing_prediction(self):
        c = case(); a = annotation(c); c['prediction']=None; a['case_hash']=case_hash(c); a['claims']=[]
        with self.assertRaises(ValueError): validate_annotation(c,a)

    def test_reject_duplicate_atomic_claims(self):
        c = case(); a = annotation(c)
        a['claims'].append(dict(a['claims'][0],id='c2'))
        with self.assertRaises(ValueError): validate_annotation(c,a)

    def test_unsupported_missing_demographic_in_annotated_labels(self):
        c = case('Patient: Hello.','The patient is 42 years old.')
        s=score([c],[annotation(c,'unsupported')])['synthetic']['full_source']
        self.assertEqual(s['unsupported_claims']['numerator'],1)
        self.assertEqual(s['age_contradictions']['numerator'],0)

    @unittest.skipUnless(Path('MTS-Dialog/Main-Dataset/MTS-Dialog-ValidationSet.csv').exists(), 'needs the MTS-Dialog submodule')
    def test_real_sample_parity_with_historical_preprocessing(self):
        # Read validation only and extract historical preprocessing without heavy imports.
        import ast
        import pandas as pd
        tree=ast.parse(Path('eval_notebook_protocol.py').read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preprocess_data' or
               isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SECTION_ORDER' for t in n.targets)]
        namespace={'pd':pd}; exec(compile(ast.Module(body=nodes,type_ignores=[]),'historical-preprocessing','exec'),namespace)
        raw=pd.read_csv('MTS-Dialog/Main-Dataset/MTS-Dialog-ValidationSet.csv')
        historical=namespace['preprocess_data'](raw).sample(n=100,random_state=42)
        new=sample_cases()
        self.assertEqual([c['source']['dialogue_id'] for c in new],historical.dialogue_id.tolist())
        self.assertEqual([c['reference'] for c in new],historical.target_text.tolist())

class GuardDiagnostics(unittest.TestCase):
    def test_blind_plan_survives_generation_but_not_source_change(self):
        c = case(); c.update(requested_section='HPI', reference='age')
        c['prediction']=None; c['model_input']=None
        plan=blind_plans([c])
        c['prediction']='The patient is 42 years old.'
        with tempfile.TemporaryDirectory() as d:
            write_queue([c],Path(d)/'queue',plan)
            c['dialogue']='Patient: Changed evidence.'
            with self.assertRaises(ValueError): write_queue([c],Path(d)/'stale',plan)

    def test_scoring_rejects_mixed_protocols_and_item_type_omissions(self):
        a = case(); a['protocol']='flask-note'; b=case();b.update(case_id='synthetic:2',protocol='notebook')
        with self.assertRaises(ValueError): score([a,b],[])
        ann=annotation(a);ann['claims'][0]['source_spans']=None
        with self.assertRaises(ValueError): validate_annotation(a,ann)

    def test_present_fact_requires_prediction_evidence(self):
        c=case(); a=annotation(c)
        a['expected_facts']=[dict(a['claims'][0],label='present')]
        del a['expected_facts'][0]['generated_span']
        with self.assertRaises(ValueError): validate_annotation(c,a)

    def test_unsupported_claim_full_vs_visible_views(self):
        c=case('Patient: Hello.','Takes aspirin.');a=annotation(c,'unsupported','medication')
        a['claims'][0]['visibility']='unclear'
        s=score([c],[a])['synthetic']
        self.assertEqual(s['full_source']['unsupported_claims']['numerator'],1)
        self.assertEqual(s['model_visible']['unsupported_claims']['denominator'],0)

class FixtureDiagnostics(unittest.TestCase):
    def test_known_synthetic_reference_set(self):
        from synthetic_fixtures import fixtures
        cases,annotations=fixtures()
        s=score(cases,annotations)
        self.assertEqual(s['cases'],17)
        self.assertEqual(s['synthetic']['full_source']['expected_fact_omissions']['numerator'],3)
        self.assertEqual(s['synthetic']['model_visible']['expected_fact_omissions']['numerator'],2)
        self.assertEqual(s['synthetic']['truncated_expected_facts'],1)
        self.assertEqual(s['synthetic']['unclear_claims'],1)
        self.assertEqual(s['human']['cases_with_annotations'],0)

class PlanningAttributionDiagnostics(unittest.TestCase):
    def test_agent_expected_fact_plan_is_not_human_review(self):
        c=case('Patient: I am 42 years old.','The patient feels well.')
        c.update(dataset_kind='real',requested_section='HPI',reference='Age 42.')
        p=blind_plans([c])[0]
        p.update(expected_facts_frozen=True,expected_facts_blind=True,expected_facts_complete=True,
          expected_facts_reviewer='Codex agent',expected_facts_reviewer_kind='agent',
          expected_facts_frozen_at='2026-10-17T00:00:00Z',expected_facts=[{
            'id':'f1','text':'Age 42','fact_type':'age','label':'omitted','visibility':'visible',
            'source_spans':[{'start':9,'end':27,'text':'I am 42 years old.'}],
            'visible_spans':[{'start':9,'end':27,'text':'I am 42 years old.'}], 'rationale':'Known fixture.'}])
        with tempfile.TemporaryDirectory() as d:
            write_queue([c],Path(d)/'queue',[p])
            a=read_jsonl(Path(d)/'queue'/'annotations.jsonl')
            s=score([c],a)
            self.assertEqual(s['human']['cases_with_annotations'],0)
            self.assertEqual(s['agent']['cases_with_annotations'],1)

    def test_mixed_review_kind_expected_facts_rejected(self):
        c=case();a=annotation(c)
        a['expected_facts_reviewer_kind']='model'
        with self.assertRaises(ValueError): validate_annotation(c,a)

class RequestProvenanceDiagnostics(unittest.TestCase):
    def test_prompt_tied_to_actual_flask_request(self):
        from factuality.generate import validate_case_request
        from factuality.io import digest
        c=case('Patient: I have pain.','Reports pain.')
        c['requested_section']='chief complaint'
        c['prompt']='Summarize the following doctor-patient dialogue into a detailed chief complaint clinical note: Patient: I have pain.'
        c['source']={'dialogue_sha256':digest(c['dialogue'])}
        self.assertEqual(validate_case_request(c),c['prompt'])
        c['prompt']=c['prompt'].replace('have pain','deny pain')
        with self.assertRaises(ValueError): validate_case_request(c)

if __name__=='__main__': unittest.main()
