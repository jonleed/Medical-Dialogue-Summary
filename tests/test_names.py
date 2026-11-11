"""Names by purpose: no version labels or run dates in tracked names; recorded hashes unchanged."""
import re
import subprocess
import unittest


class Names(unittest.TestCase):
    def test_rubric_and_schema_hashes_unchanged(self):
        from factuality.judge import rubric_hash, schema_hash
        self.assertEqual(rubric_hash(), 'b6031aa72a9adcea0a87d44b7a77ea38faab986ce697a12a173a4a87cfbe2768')
        self.assertEqual(schema_hash(), '9e7fb29fd42cb441c41262c21bc5fa22e8c2b49688fc091be3fc42794590421d')

    def test_no_version_or_date_in_tracked_names(self):
        names = subprocess.run(['git', 'ls-files', 'factuality', 'docs', 'results/factuality', 'tests'],
                               capture_output=True, text=True, check=True).stdout.split()
        bad = [n for n in names
               if re.search(r'(?<![a-z])v[23](?![0-9])|20\d\d-\d\d', n)]
        self.assertEqual(bad, [])


if __name__ == '__main__':
    unittest.main()
