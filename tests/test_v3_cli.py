"""v3 CLI helpers that pick judge output folders."""
import tempfile
import unittest
from pathlib import Path

from factuality.v3_cli import latest_round_dir, round_split


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
