"""Benchmark bundle export: model-named, opaque IDs resolved, no ledgers or token IDs."""
import json
import tempfile
import unittest
from pathlib import Path

from factuality.benchmark import FILES, export_bundle
from factuality.io import read_jsonl


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def case(cid, prediction):
    d = 'Patient: I am 40 years old.'
    return {'case_id': cid, 'dialogue': d, 'requested_section': 'History of Present Illness', 'reference': 'ref',
            'source': {'split': 'val', 'dialogue_id': cid[4:], 'dialogue_sha256': 'h'}, 'prediction': prediction,
            'model_input': {'text': d, 'truncated': False, 'full_token_count': 9, 'input_ids': [1, 2], 'attention_mask': [1, 1],
                            'sha256': 's'}, 'provenance': {'base_revision': 'r1', 'adapter_weights_sha256': 'a1'},
            'protocol': 'flask-note'}


class Export(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        local, v3 = self.root / 'local', self.root / 'local' / 'v3'
        self.runs = {'run-a-2026': 'model-a', 'run-b-2026': 'model-b'}
        for run in self.runs:
            write_jsonl(local / run / 'cases.jsonl', [case('val:1', 'The patient is 40.'), case('val:2', 'Cough.')])
        write_jsonl(v3 / 'plans' / 'plans.jsonl', [{'case_id': 'val:1', 'expected_facts': []}])
        write_jsonl(v3 / 'annotations.jsonl', [{'case_id': 'val:1', 'run': 'run-a-2026', 'claims': [], 'expected_facts': []}])
        (v3 / 'sample.json').write_text(json.dumps({'calibration': ['val:1'], 'heldout': ['val:2'], 'random': ['val:1']}))
        write_jsonl(v3 / 'cf_cases_v2.jsonl', [dict(case('cf:age_shift:val:1', None), counterfactual={'family': 'age_shift', 'gold': {'age': 47}})])
        write_jsonl(v3 / 'cf-a' / 'cases.jsonl', [case('cf:age_shift:val:1', 'The patient is 47.')])
        (v3 / 'judge_manifest.json').write_text(json.dumps({'freeze': {'model': 'm', 'rubric_hash': 'x'}, 'probe': {'passed': True},
                                                            'terms_confirmed_by_owner': True}))
        jd = v3 / 'judge' / 'heldout'
        (jd / 'abc').mkdir(parents=True)
        (jd / 'key.json').write_text(json.dumps({'abc': {'run': 'run-b-2026', 'case_id': 'val:2'}}))
        (jd / 'abc' / 't1.json').write_text(json.dumps({'opaque_id': 'abc', 'trial': 1, 'status': 'complete', 'raw': {'big': 1},
                                                        'result': {'claims': []}, 'cli_version': 'v', 'rubric_hash': 'x'}))
        (v3 / 'judge_ledger.json').write_text('{"entries": {}}')
        self.out = self.root / 'bundle'
        export_bundle(local, self.out, self.runs, cf_runs={'cf-a': 'model-a'})

    def test_files_written(self):
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), sorted(FILES))

    def test_model_names_replace_run_names(self):
        text = ''.join(p.read_text() for p in self.out.iterdir())
        self.assertNotIn('run-a-2026', text)
        self.assertNotIn('run-b-2026', text)
        self.assertEqual({r['model'] for r in read_jsonl(self.out / 'summaries.jsonl')}, {'model-a', 'model-b'})
        self.assertEqual(read_jsonl(self.out / 'labels.jsonl')[0]['model'], 'model-a')

    def test_cases_deduplicated_without_predictions_or_token_ids(self):
        cases = read_jsonl(self.out / 'cases.jsonl')
        self.assertEqual([c['case_id'] for c in cases], ['val:1', 'val:2'])
        self.assertNotIn('prediction', cases[0])
        text = (self.out / 'summaries.jsonl').read_text()
        self.assertNotIn('input_ids', text)
        self.assertNotIn('attention_mask', text)

    def test_judge_outputs_resolved_and_trimmed(self):
        [j] = read_jsonl(self.out / 'judge_outputs.jsonl')
        self.assertEqual((j['model'], j['case_id'], j['split'], j['trial']), ('model-b', 'val:2', 'heldout', 1))
        self.assertNotIn('raw', j)
        self.assertNotIn('opaque_id', j)

    def test_no_ledger_or_terms(self):
        self.assertFalse((self.out / 'judge_ledger.json').exists())
        self.assertNotIn('terms_confirmed', (self.out / 'judge_config.json').read_text())

    def test_counterfactuals(self):
        self.assertEqual(read_jsonl(self.out / 'counterfactual_summaries.jsonl')[0]['model'], 'model-a')
        self.assertEqual(read_jsonl(self.out / 'counterfactual_cases.jsonl')[0]['counterfactual']['gold'], {'age': 47})

    def test_refuses_existing_output(self):
        with self.assertRaises(ValueError):
            export_bundle(self.root / 'local', self.out, self.runs, cf_runs={'cf-a': 'model-a'})


if __name__ == '__main__':
    unittest.main()


class ExportFromConfig(Export):
    def test_export_from_config_writes_bundle(self):
        from factuality.benchmark import export_from_config
        from factuality.config import load_config
        cfg = self.root / 'factuality.toml'
        cfg.write_text('''
[paths]
benchmark = "b"
work = "local"
published = "pub"
study = "local/v3"
counterfactual_packet = "local/v3/cf_cases_v2.jsonl"
[models.model-a]
generator = "command"
command = ["x"]
run_dir = "run-a-2026"
counterfactual_run_dir = "cf-a"
[models.model-b]
generator = "command"
command = ["x"]
run_dir = "run-b-2026"
''')
        out = export_from_config(load_config(cfg), self.root / 'bundle2')
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(FILES))
        self.assertEqual({r['model'] for r in read_jsonl(out / 'summaries.jsonl')}, {'model-a', 'model-b'})
