"""factuality.toml: models, pinned revisions and paths live in config, not code."""
import tempfile
import unittest
from pathlib import Path

from factuality.config import ROOT, ConfigError, load_config

BASE = '''
[paths]
benchmark = "data/b"
work = "w"
published = "p"
study = "w/study"
counterfactual_packet = "w/study/packet.jsonl"
'''


def write(text):
    d = Path(tempfile.mkdtemp())
    (d / 'factuality.toml').write_text(text)
    return d / 'factuality.toml'


class RepoConfig(unittest.TestCase):
    def test_repo_models_and_revisions(self):
        c = load_config()
        self.assertEqual(set(c.models), {'biobart-large-lora', 'bart-base-lora', 'bart-base'})
        self.assertEqual(c.models['biobart-large-lora'].base_revision, '47c968d3aba260d5737fda6f5b2ad3aa35ce75e8')
        self.assertEqual(c.models['bart-base-lora'].base_revision, 'aadd2ab0ae0c8268c7c9693540e9904811f36177')
        self.assertEqual(c.models['bart-base-lora'].adapter_revision, '8a61695044124b479ba73da5134f064b5e36ae53')
        self.assertEqual((c.models['bart-base'].generator, c.models['bart-base'].adapter), ('flask', None))
        self.assertEqual(c.benchmark, ROOT / 'data' / 'factuality_benchmark')


class Validation(unittest.TestCase):
    def test_missing_revision_names_model(self):
        path = write(BASE + '[models.mine]\ngenerator = "hf"\nbase_model = "x/y"\n')
        with self.assertRaises(ConfigError) as e:
            load_config(path)
        self.assertIn('mine', str(e.exception))

    def test_unknown_generator(self):
        with self.assertRaises(ConfigError):
            load_config(write(BASE + '[models.m]\ngenerator = "magic"\n'))

    def test_command_requires_command(self):
        with self.assertRaises(ConfigError):
            load_config(write(BASE + '[models.m]\ngenerator = "command"\n'))

    def test_command_model_ok(self):
        c = load_config(write(BASE + '[models.m]\ngenerator = "command"\ncommand = ["python", "s.py"]\n'))
        self.assertEqual(c.models['m'].command, ['python', 's.py'])
        self.assertEqual(c.models['m'].label, 'm')

    def test_relative_paths_resolve_against_config_dir(self):
        path = write(BASE)
        c = load_config(path)
        self.assertEqual(c.work, path.parent / 'w')
        self.assertEqual(c.counterfactual_packet, path.parent / 'w' / 'study' / 'packet.jsonl')


if __name__ == '__main__':
    unittest.main()
