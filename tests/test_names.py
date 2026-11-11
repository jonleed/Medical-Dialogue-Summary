"""Names by purpose: no version labels or run dates in tracked names; recorded hashes unchanged."""
import os
import re
import subprocess
from pathlib import Path
import unittest


class Names(unittest.TestCase):
    def test_rubric_and_schema_hashes_unchanged(self):
        from factuality.judge import rubric_hash, schema_hash
        self.assertEqual(rubric_hash(), 'b6031aa72a9adcea0a87d44b7a77ea38faab986ce697a12a173a4a87cfbe2768')
        self.assertEqual(schema_hash(), '9e7fb29fd42cb441c41262c21bc5fa22e8c2b49688fc091be3fc42794590421d')

    def test_no_version_or_date_in_tracked_names(self):
        names = subprocess.run(['git', 'ls-files', 'factuality', 'docs', 'results/factuality', 'tests'],
                               capture_output=True, text=True, check=True).stdout.split()
        bad = [n for n in names if not n.startswith('docs/superpowers/')
               and re.search(r'(?<![a-z])v[23](?![0-9])|20\d\d-\d\d', n)]
        self.assertEqual(bad, [])

    def test_docs_present_tense_and_links_resolve(self):
        tracked = set(subprocess.run(['git', 'ls-files'], capture_output=True, text=True, check=True).stdout.split())
        docs = ['README.md'] + sorted(n for n in tracked if re.fullmatch(r'docs/factuality_[a-z_]+\.md', n))
        dated = re.compile(r'\b(v[23]|pilot|superseded|2026-\d\d-\d\d|October 2026|November 2026)\b', re.I)
        problems = []
        for doc in docs:
            text = Path(doc).read_text()
            problems += [f'{doc}: {m.group(0)!r}' for m in dated.finditer(text)]
            for link in re.findall(r'\]\(([^)#\s]+)', text):
                if '://' in link:
                    continue
                target = os.path.normpath(os.path.join(os.path.dirname(doc), link))
                if target not in tracked and not any(t.startswith(target.rstrip('/') + '/') for t in tracked):
                    problems.append(f'{doc}: broken link {link}')
        self.assertEqual(problems, [])

    def test_cli_help_has_no_version_or_dates(self):
        from factuality.cli import STAGES, build_parser
        parser = build_parser()
        texts = [parser.format_help()]
        subs = next(a for a in parser._actions if a.__class__.__name__ == '_SubParsersAction')
        texts += [sp.format_help() for sp in subs.choices.values()]
        texts += [h for _, h in STAGES.values()]
        bad = [t for t in texts if re.search(r'(?<![a-z_])v[23](?![0-9])|<date>|20\d\d-\d\d', t)]
        self.assertEqual(bad, [])


if __name__ == '__main__':
    unittest.main()
