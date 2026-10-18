import json
import tempfile
import unittest
from pathlib import Path

class LocalFactsTests(unittest.TestCase):
    def test_family_age_is_not_patient_age(self):
        from factuality.facts import extract_facts
        facts=extract_facts('Patient: I am 57 years old. My father is 67.',dialogue=True)
        ages=[f['value'] for f in facts if f['subject']=='patient' and f['fact_type']=='age']
        self.assertIn(57,ages);self.assertNotIn(67,ages)

    def test_reference_cannot_supply_missing_dialogue_evidence(self):
        from factuality.facts import freeze_expected
        with self.assertRaises(ValueError):
            freeze_expected('Patient: I cough.','chief complaint','Diabetes.',[{'id':'f1','quote':'diabetes','text':'Diabetes','fact_type':'diagnosis','required':True}])

    def test_optional_fact_stays_optional(self):
        from factuality.facts import freeze_expected
        plan=freeze_expected('Patient: I am 57.','chief complaint','Age 57.',[{'id':'f','quote':'I am 57.','text':'Age 57','fact_type':'age','required':False}])
        self.assertFalse(plan['facts'][0]['required'])

class ProtocolTests(unittest.TestCase):
    def test_blinded_payload_and_balanced_cases(self):
        from factuality.free_pilot_cases import build_cases
        from factuality.free_protocol import make_payload
        cases=build_cases();self.assertEqual(len(cases),24)
        for cap in ['age','gender','unsupported','omission']:
            group=[c for c in cases if c['gold']['capability']==cap]
            self.assertEqual(sum(c['gold']['error'] for c in group),3)
            self.assertEqual(len(group),6)
        payload=make_payload(cases[0]);self.assertNotIn('gold',payload)
        self.assertNotIn('target_span',json.dumps(payload))

    def test_fabricated_source_quote_fails(self):
        from factuality.free_protocol import validate_response
        payload={'source':'Patient is 57.','prediction':'Patient is 67.','plans':[]}
        raw={'claims_complete':True,'claims':[{'text':'Age 67','fact_type':'age','generated_quote':'67','label':'contradicted','rationale':'age mismatch','evidence_quotes':['Patient is 47.']}],'expected_facts':[]}
        with self.assertRaises(ValueError):validate_response(payload,raw)

    def test_repeated_evidence_stays_ambiguous(self):
        from factuality.free_protocol import validate_response
        p={'source':'Doctor: pain. Patient: pain.','prediction':'Pain.','plans':[]}
        raw={'claims_complete':True,'claims':[{'text':'Pain','fact_type':'symptom','generated_quote':'Pain','label':'supported','rationale':'documented','evidence_quotes':['pain']}],'expected_facts':[]}
        result=validate_response(p,raw)
        self.assertEqual(result['claims'][0]['label'],'unclear')
        self.assertEqual(len(result['claims'][0]['source_spans']),2)

class BudgetTests(unittest.TestCase):
    def test_limit_and_restart_do_not_allow_extra_dispatch(self):
        from factuality.opencode_backend import BudgetLedger,MODEL
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ledger.json';ledger=BudgetLedger(path)
            ledger.reserve('job-0',MODEL,100)
            with self.assertRaises(ValueError):BudgetLedger(path).reserve('job-0',MODEL,100)
            ledger.settle('job-0',{'input':10,'output':20},'complete')
            for i in range(1,24):
                ledger.reserve('job-'+str(i),MODEL,100);ledger.settle('job-'+str(i),{'input':10,'output':20},'complete')
            with self.assertRaises(ValueError):ledger.reserve('job-24',MODEL,100)
            with self.assertRaises(ValueError):ledger.reserve('paid','openai/gpt-6-astra',100)

    def test_unsettled_reservation_survives_restart(self):
        from factuality.opencode_backend import BudgetLedger,MODEL
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ledger.json';BudgetLedger(p).reserve('lost',MODEL,250000)
            with self.assertRaises(ValueError):BudgetLedger(p).reserve('next',MODEL,100)

class ReportingTests(unittest.TestCase):
    def test_missing_positive_is_a_false_negative(self):
        from factuality.free_report import binary_metrics
        m=binary_metrics([(True,True),(True,None),(False,True),(False,False)])
        self.assertEqual((m['tp'],m['fn'],m['fp'],m['tn']),(1,1,1,1))
        self.assertEqual(m['recall'],.5);self.assertEqual(m['precision'],.5)

if __name__=='__main__':unittest.main()

class AdditionalControls(unittest.TestCase):
    def test_output_can_be_present_through_a_paraphrase(self):
        from factuality.free_protocol import validate_response
        from factuality.facts import freeze_expected
        plan=freeze_expected('Patient: I am allergic to penicillin.','allergies','',[{'id':'f1','quote':'allergic to penicillin','text':'Penicillin allergy','fact_type':'other','required':True}])
        p={'source':plan['dialogue'],'prediction':'Penicillin allergy.','plans':plan['facts']}
        r={'claims_complete':True,'claims':[{'text':'Penicillin allergy','fact_type':'other','generated_quote':'Penicillin allergy','label':'supported','rationale':'same meaning','evidence_quotes':['allergic to penicillin']}],'expected_facts':[{'id':'f1','label':'present','generated_quote':'Penicillin allergy','rationale':'same meaning'}]}
        self.assertEqual(validate_response(p,r)['expected_facts'][0]['label'],'present')
        r['expected_facts']=[]
        with self.assertRaises(ValueError):validate_response(p,r)

    def test_optional_omission_is_excluded(self):
        from factuality.free_protocol import make_payload,validate_response
        from factuality.free_pilot_cases import build_cases
        p=make_payload(build_cases()[-2])
        r={'claims_complete':True,'claims':[{'text':'Cough','fact_type':'symptom','generated_quote':'Cough','label':'supported','rationale':'source cough','evidence_quotes':['cough']}],'expected_facts':[{'id':'f1','label':'omitted','generated_quote':'','rationale':'age not in note'}]}
        self.assertEqual(validate_response(p,r)['expected_facts'][0]['label'],'excluded')

    def test_unknown_usage_retains_full_reservation(self):
        from factuality.opencode_backend import BudgetLedger,MODEL
        with tempfile.TemporaryDirectory() as d:
            ledger=BudgetLedger(Path(d)/'ledger.json')
            ledger.reserve('j',MODEL,250000);ledger.settle('j',None,'complete')
            with self.assertRaises(ValueError):ledger.reserve('next',MODEL,1)

    def test_deadline_terminates_the_owned_process(self):
        import subprocess,sys,time,os
        from factuality.opencode_backend import run_process
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pid';code='import os,time;open('+repr(str(p))+',"w").write(str(os.getpid()));time.sleep(30)'
            with self.assertRaises(subprocess.TimeoutExpired):run_process([sys.executable,'-c',code],d,.2)
            pid=int(p.read_text())
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    def test_failed_control_is_not_a_true_negative(self):
        from factuality.free_report import binary_metrics
        m=binary_metrics([(False,None)]);self.assertEqual(m['tn'],0);self.assertEqual(m['unassessed'],1)

    def test_wrong_type_label_does_not_hide_extraction_miss(self):
        from factuality.free_report import assess
        from factuality.free_pilot_cases import build_cases
        c=build_cases()[0]
        response={'status':'complete','result':{'claims':[{'fact_type':'quantity','label':'contradicted','generated_span':c['gold']['target_span']}],'expected_facts':[]}}
        result=assess(c,response);self.assertFalse(result['extracted']);self.assertIsNone(result['prediction'])

class RuntimePolicyTests(unittest.TestCase):
    def test_extra_deny_rule_is_safe_but_later_allow_is_not(self):
        from factuality.opencode_backend import tools_denied
        deny={'action':'*','resource':'*','effect':'deny'}
        self.assertTrue(tools_denied([deny,{'action':'browser','resource':'*','effect':'deny'}]))
        self.assertFalse(tools_denied([deny,{'action':'read','resource':'*','effect':'allow'}]))

class StopOnProviderRejectionTests(unittest.TestCase):
    def test_provider_block_survives_restart(self):
        from factuality.opencode_backend import BudgetLedger,MODEL
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ledger.json';ledger=BudgetLedger(p);ledger.block('provider denied free-tier access')
            with self.assertRaises(ValueError):BudgetLedger(p).reserve('next',MODEL,1000)
            self.assertEqual(BudgetLedger(p).read()['entries'],{})

class EvidenceIntegrityTests(unittest.TestCase):
    def test_optional_correction_preserves_model_proposal(self):
        from factuality.free_protocol import make_payload,validate_response
        from factuality.free_pilot_cases import build_cases
        p=make_payload(build_cases()[-2])
        raw={'claims_complete':True,'claims':[{'text':'Cough','fact_type':'symptom','generated_quote':'Cough','label':'supported','rationale':'source cough','evidence_quotes':['cough']}],'expected_facts':[{'id':'f1','label':'omitted','generated_quote':'','rationale':'age not stated'}]}
        fact=validate_response(p,raw)['expected_facts'][0]
        self.assertEqual(fact['label'],'excluded');self.assertEqual(fact['proposed_label'],'omitted')

    def test_string_false_cannot_become_required(self):
        from factuality.facts import freeze_expected
        with self.assertRaises(ValueError):freeze_expected('Patient: age 57.','chief complaint','',[{'id':'f','quote':'age 57','text':'age57','fact_type':'age','required':'false'}])

    def test_backend_itself_persists_provider_denial(self):
        from contextlib import contextmanager
        from unittest.mock import patch
        from factuality.opencode_backend import run_review,BudgetLedger,MODEL
        @contextmanager
        def isolated(*args):yield 'http://127.0.0.1:1',{},lambda path:{}
        with tempfile.TemporaryDirectory() as tmp:
            ledger=BudgetLedger(Path(tmp)/'ledger.json')
            job={'id':'denied','version':'test','payload':{'source':'Patient: fever.','prediction':'Fever.','plans':[]}}
            events='{"type":"error","error":{"message":"OpenCode free tier restriction","type":"provider.auth"}}\n'
            with patch('factuality.opencode_backend.private_server',isolated),patch('factuality.opencode_backend.run_process',return_value=(1,events,'')):
                self.assertEqual(run_review(job,MODEL,ledger,Path(tmp)/'review')['status'],'failed')
            with self.assertRaises(ValueError):ledger.reserve('another',MODEL,100)
