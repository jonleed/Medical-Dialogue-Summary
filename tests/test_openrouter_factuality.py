import copy
import json
import tempfile
import unittest
from pathlib import Path


class FreeRoutingTests(unittest.TestCase):
    def test_paid_or_unpriced_endpoint_is_rejected(self):
        from factuality.openrouter_pilot import verify_endpoint, MODEL
        endpoint={'tag':'nvidia','model_id':MODEL,'pricing':{'prompt':'0','completion':'0'},'supported_parameters':['max_tokens','temperature','reasoning']}
        self.assertEqual(verify_endpoint({'id':MODEL,'endpoints':[endpoint]})['tag'],'nvidia')
        for prices in [{'prompt':'0.001','completion':'0'},{'prompt':'0'},{'prompt':'0','completion':'0','request':'1'}]:
            bad=copy.deepcopy(endpoint);bad['pricing']=prices
            with self.assertRaises(ValueError):verify_endpoint({'id':MODEL,'endpoints':[bad]})
        with self.assertRaises(ValueError):verify_endpoint({'id':MODEL.replace(':free',''),'endpoints':[endpoint]})

    def test_request_has_server_enforced_zero_prices_and_no_fallback(self):
        from factuality.openrouter_pilot import request_body, MODEL
        body=request_body({'source':'Patient: cough.','prediction':'Cough.','plans':[]})
        self.assertEqual(body['model'],MODEL)
        self.assertNotIn('models',body)
        self.assertFalse(body['provider']['allow_fallbacks'])
        self.assertEqual(body['provider']['only'],['nvidia'])
        self.assertEqual(body['provider']['max_price'],{'prompt':0,'completion':0,'request':0})
        self.assertNotIn('tools',body)

    def test_free_ledger_retains_prior_attempts_in_cap(self):
        from factuality.openrouter_pilot import RouterLedger, MODEL
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'ledger.json'
            p.write_text(json.dumps({'entries':{f'old-{i}':{'status':'failed','input_bound':10000,'output_bound':2000} for i in range(5)}}))
            ledger=RouterLedger(p)
            for i in range(19):
                ledger.reserve(str(i),MODEL,10000);ledger.settle(str(i),{'input':100,'output':100},'complete')
            with self.assertRaises(ValueError):ledger.reserve('extra',MODEL,10000)
            with self.assertRaises(ValueError):RouterLedger(Path(tmp)/'paid.json').reserve('paid','nvidia/nemotron-3-ultra-550b-a55b',10000)

    def test_response_cost_tools_and_truncation_are_rejected(self):
        from factuality.openrouter_pilot import parse_response, MODEL
        x={'model':MODEL.replace(':free',''),'provider':'Nvidia','usage':{'prompt_tokens':80,'completion_tokens':90,'cost':0},'choices':[{'finish_reason':'stop','message':{'content':'{"claims_complete":true,"claims":[],"expected_facts":[]}'}}]}
        raw,usage=parse_response(x)
        self.assertEqual(usage,{'input':80,'output':90});self.assertTrue(raw['claims_complete'])
        for transform in [lambda z:z['usage'].update(cost=.01),lambda z:z['choices'][0].update(finish_reason='length'),lambda z:z['choices'][0]['message'].update(tool_calls=[{'id':'x'}]),lambda z:z.update(model='another/model')]:
            bad=copy.deepcopy(x);transform(bad)
            with self.assertRaises(ValueError):parse_response(bad)

    def test_report_labels_the_actual_reviewer(self):
        from factuality.free_report import build_report
        from factuality.openrouter_pilot import MODEL
        self.assertEqual(build_report([],[],{'entries':{}},model=MODEL)['model'],MODEL)

class InvalidEvidenceTests(unittest.TestCase):
    def test_source_only_quote_cannot_prove_summary_coverage(self):
        from factuality.openrouter_pilot import validate_completion
        payload={'source':'Doctor: Return tomorrow.','prediction':'Follow-up discussed.','plans':[{'id':'f1','required':True}]}
        raw={'claims_complete':True,'claims':[{'text':'Follow-up','fact_type':'other','generated_quote':'Follow-up','label':'supported','rationale':'source follow-up','evidence_quotes':['Return tomorrow']}],'expected_facts':[{'id':'f1','label':'present','generated_quote':'Return tomorrow','rationale':'covered'}]}
        result=validate_completion(payload,raw)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['failure_stage'],'evidence_validation')
        self.assertNotIn('result',result)

class PortablePilotTests(unittest.TestCase):
    def test_prepare_without_private_history_preserves_attempts_on_restart(self):
        from unittest.mock import patch
        import factuality.openrouter_pilot as pilot
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'run'
            with patch.object(pilot,'ORIGINAL',root/'missing-history'):
                pilot.prepare(out)
                ledger=pilot.RouterLedger(out/'ledger.json')
                ledger.reserve('j',pilot.MODEL,100);ledger.settle('j',{'input':10,'output':10},'failed')
                pilot.prepare(out)
            manifest,cases=pilot.frozen(out)
            self.assertEqual(len(cases),24)
            self.assertEqual(manifest['prior_attempts'],0)
            self.assertEqual(len(ledger.read()['entries']),1)
            manifest['selected_case_ids'].append('case-02')
            (out/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):pilot.frozen(out)

    def test_custom_report_cannot_overwrite_published_results(self):
        from unittest.mock import patch
        import factuality.openrouter_pilot as pilot
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'custom';public=root/'published/openrouter-pilot-2026-10-18'
            public.mkdir(parents=True);marker=public/'metrics.json';marker.write_text('{"preserved":true}')
            with patch.object(pilot,'ORIGINAL',root/'missing-history'),patch.object(pilot,'OUT',root/'published/local/openrouter-pilot-2026-10-18'):
                pilot.prepare(out)
                metrics=pilot.report(out)
            self.assertEqual(marker.read_text(),'{"preserved":true}')
            self.assertEqual(metrics['valid_reviews'],0)
            self.assertEqual(metrics['attempts'],0)
