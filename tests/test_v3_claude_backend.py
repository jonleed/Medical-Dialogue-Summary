"""Claude Code judge backend: isolation flags, failure classes, ledger, probe and run gates (fake CLI only)."""
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from factuality.claude_code_backend import (CallLedger, MODEL, build_command, call, cli_version, isolation_probe,
                                            require_run_preconditions)

FAKE = str(Path(__file__).resolve().parent / 'fixtures' / 'fake_claude.py')
SCHEMA = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']}


class Env:
    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.values}
        os.environ.update(self.values)

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Backend(unittest.TestCase):
    def setUp(self):
        os.chmod(FAKE, os.stat(FAKE).st_mode | stat.S_IXUSR)
        self.tmp = Path(tempfile.mkdtemp())
        self.ledger = CallLedger(self.tmp / 'ledger.json', cap=10)

    def run_call(self, mode, job='j1', timeout=2):
        with Env(FAKE_CLAUDE_MODE=mode, FAKE_CLAUDE_STATE=str(self.tmp / f'{job}.state'),
                 FAKE_CLAUDE_STDIN_LOG=str(self.tmp / 'stdin.log')):
            return call({'case_id': 'x1'}, exe=FAKE, ledger=self.ledger, job_id=job, timeout=timeout,
                        system_prompt='rubric', schema=SCHEMA)

    def test_command_flags(self):
        cmd = build_command('claude', 'rubric', SCHEMA)
        self.assertEqual(cmd[:4], ['claude', '-p', '--model', MODEL])
        for flag in ['--strict-mcp-config', '--disable-slash-commands', '--no-session-persistence']:
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index('--tools') + 1], '')
        self.assertEqual(cmd[cmd.index('--setting-sources') + 1], 'project')
        self.assertEqual(json.loads(cmd[cmd.index('--json-schema') + 1]), SCHEMA)
        self.assertNotIn('--bare', cmd)

    def test_success_structured_output(self):
        r = self.run_call('success')
        self.assertEqual((r['status'], r['raw'], r['attempts']), ('complete', {'ok': True}, 1))
        self.assertEqual(r['usage']['output_tokens'], 120)
        self.assertTrue(self.ledger.done('j1'))

    def test_payload_on_stdin_in_empty_dir(self):
        self.run_call('success')
        [log] = [json.loads(l) for l in (self.tmp / 'stdin.log').read_text().splitlines()]
        self.assertIn('"case_id": "x1"', log['stdin'])
        self.assertNotEqual(Path(log['cwd']).resolve(), Path.cwd().resolve())
        self.assertFalse(Path(log['cwd']).exists())  # scratch dir removed after the call

    def test_result_text_fallback(self):
        self.assertEqual(self.run_call('result_text')['raw'], {'ok': True})

    def test_invalid_json_is_content_failure_without_retry(self):
        r = self.run_call('invalid_json')
        self.assertEqual((r['status'], r['attempts']), ('content_failed', 1))

    def test_timeout_then_success_retries_once(self):
        r = self.run_call('timeout_then_success', timeout=1)
        self.assertEqual((r['status'], r['attempts']), ('complete', 2))

    def test_two_timeouts_fail(self):
        r = self.run_call('timeout', timeout=1)
        self.assertEqual((r['status'], r['attempts']), ('transport_failed', 2))

    def test_crash_then_success(self):
        self.assertEqual(self.run_call('crash_then_success')['attempts'], 2)

    def test_usage_limit_blocks_ledger(self):
        r = self.run_call('usage_limit')
        self.assertEqual((r['status'], r['attempts']), ('usage_limited', 1))
        with self.assertRaises(ValueError):
            self.ledger.reserve('j2')

    def test_not_logged_in_blocks_ledger(self):
        self.assertEqual(self.run_call('not_logged_in')['status'], 'auth_failed')
        with self.assertRaises(ValueError):
            self.ledger.reserve('j2')

    def test_completed_job_not_resent(self):
        self.run_call('success')
        with self.assertRaises(ValueError):
            self.ledger.reserve('j1')

    def test_cap(self):
        ledger = CallLedger(self.tmp / 'cap.json', cap=2)
        for job in ['a', 'b']:
            ledger.reserve(job)
            ledger.settle(job, 'complete', None, attempts=1)
        with self.assertRaises(ValueError):
            ledger.reserve('c')


class Probe(unittest.TestCase):
    def probe(self, mode):
        with Env(FAKE_CLAUDE_MODE=mode):
            return isolation_probe(FAKE, timeout=5)

    def test_clean(self):
        p = self.probe('probe_clean')
        self.assertTrue(p['passed'], p['failures'])
        self.assertEqual(p['cli_version'], '2.1.311 (Claude Code)')

    def test_hook_event_fails(self):
        self.assertFalse(self.probe('probe_hook')['passed'])

    def test_user_plugin_fails(self):
        self.assertFalse(self.probe('probe_user_plugin')['passed'])

    def test_tools_fail(self):
        self.assertFalse(self.probe('probe_tools')['passed'])


class Preconditions(unittest.TestCase):
    def test_terms_required(self):
        with self.assertRaises(ValueError):
            require_run_preconditions({'probe': {'passed': True, 'cli_version': cli_version(FAKE)}}, FAKE)

    def test_probe_must_match_version(self):
        manifest = {'terms_confirmed_by_owner': True, 'probe': {'passed': True, 'cli_version': '2.0.0'}}
        with self.assertRaises(ValueError):
            require_run_preconditions(manifest, FAKE)

    def test_ok(self):
        require_run_preconditions({'terms_confirmed_by_owner': True,
                                   'probe': {'passed': True, 'cli_version': cli_version(FAKE)}}, FAKE)


if __name__ == '__main__':
    unittest.main()
