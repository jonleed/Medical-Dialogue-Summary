"""Summary generators: external command and Hugging Face (stubbed; no downloads)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from factuality.config import ModelSpec
from factuality.generators import DECODING, generate_summaries
from factuality.io import read_jsonl

ECHO = str(Path(__file__).resolve().parent / 'fixtures' / 'echo_summarizer.py')
CASES = [{'case_id': f'val:{i}', 'dialogue': f'Patient: dialogue number {i} about a cough.', 'requested_section': 'plan'}
         for i in range(3)]


class Command(unittest.TestCase):
    def spec(self):
        return ModelSpec(name='echo', generator='command', label='echo', command=[sys.executable, ECHO])

    def test_one_record_per_case_in_order(self):
        out = Path(tempfile.mkdtemp()) / 's.jsonl'
        self.assertEqual(generate_summaries(self.spec(), CASES, out), 3)
        rows = read_jsonl(out)
        self.assertEqual([r['case_id'] for r in rows], ['val:0', 'val:1', 'val:2'])
        self.assertEqual(rows[0]['prediction'], CASES[0]['dialogue'][:40])

    def test_missing_case_raises(self):
        with patch.dict(os.environ, {'ECHO_DROP_LAST': '1'}):
            with self.assertRaises(ValueError) as e:
                generate_summaries(self.spec(), CASES, Path(tempfile.mkdtemp()) / 's.jsonl')
        self.assertIn('val:2', str(e.exception))


class StubTokenizer:
    def __call__(self, text, **kw):
        ids = list(range(1, 600 if 'long' in text else 6))
        if kw.get('return_tensors'):
            return {'input_ids': [ids[:512]], 'attention_mask': [[1] * len(ids[:512])]}
        return {'input_ids': ids}

    def decode(self, ids, skip_special_tokens=True):
        return f'decoded {len(ids)} tokens'


class StubModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kw):
        self.calls.append(kw)
        return [[7, 8, 9]]

    def eval(self):
        return self


class HF(unittest.TestCase):
    def test_revision_and_decoding(self):
        model = StubModel()
        spec = ModelSpec(name='h', generator='hf', label='h', base_model='org/model', base_revision='abc123')
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=StubTokenizer()) as tok, \
             patch('transformers.AutoModelForSeq2SeqLM.from_pretrained', return_value=model) as mod:
            out = Path(tempfile.mkdtemp()) / 's.jsonl'
            cases = CASES[:1] + [dict(CASES[1], dialogue='a long dialogue')]
            generate_summaries(spec, cases, out)
        self.assertEqual(tok.call_args.kwargs['revision'], 'abc123')
        self.assertEqual(mod.call_args.kwargs['revision'], 'abc123')
        self.assertEqual({k: model.calls[0][k] for k in DECODING}, DECODING)
        self.assertEqual(DECODING, {'do_sample': False, 'num_beams': 4, 'repetition_penalty': 2.0,
                                    'no_repeat_ngram_size': 4, 'max_length': 512})
        rows = read_jsonl(out)
        self.assertEqual(rows[0]['model_input']['truncated'], False)
        self.assertEqual(rows[1]['model_input']['truncated'], True)
        self.assertEqual(rows[0]['prediction'], 'decoded 3 tokens')


if __name__ == '__main__':
    unittest.main()
