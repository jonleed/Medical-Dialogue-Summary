"""v3 CLI helpers that pick judge output folders."""
import tempfile
import unittest
from pathlib import Path

from factuality.cli import latest_round_dir, round_split


class Rounds(unittest.TestCase):
    def test_round_split_names(self):
        self.assertEqual(round_split('calibration', 1), 'calibration')
        self.assertEqual(round_split('calibration', 2), 'calibration-r2')

    def test_latest_round(self):
        root = Path(tempfile.mkdtemp())
        for name in ['calibration', 'calibration-r2', 'calibration-r10', 'heldout']:
            (root / name).mkdir()
        self.assertEqual(latest_round_dir(root, 'calibration').name, 'calibration-r10')
        self.assertEqual(latest_round_dir(root, 'heldout').name, 'heldout')


if __name__ == '__main__':
    unittest.main()


class Terms(unittest.TestCase):
    def test_confirm_terms_records_attestation_and_keeps_probe(self):
        import json
        from factuality.claude_code_backend import confirm_terms
        path = Path(tempfile.mkdtemp()) / 'judge_manifest.json'
        path.write_text(json.dumps({'probe': {'passed': True}}))
        confirm_terms(path)
        m = json.loads(path.read_text())
        self.assertIs(m['terms_confirmed_by_owner'], True)
        self.assertTrue(m['terms_confirmed_at'])
        self.assertEqual(m['probe'], {'passed': True})

    def test_missing_terms_message_names_the_fix(self):
        from factuality.claude_code_backend import require_run_preconditions
        with self.assertRaises(ValueError) as e:
            require_run_preconditions({}, 'claude')
        self.assertIn('probe --confirm-terms', str(e.exception))
