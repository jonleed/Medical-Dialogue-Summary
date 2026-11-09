"""Isolated headless Claude Code judge calls under the owner's subscription (spec §6.5).

`--bare` cannot be used (it forces API-key auth), so isolation comes from flags
plus an empty working directory, and is verified by `isolation_probe` before any
judge run. Every call is reserved in a file-locked ledger with a hard call cap.
"""
import fcntl
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

MODEL = 'claude-sonnet-5-5'
CALL_CAP = 1000
ALLOWED_TOOLS = ('StructuredOutput',)
TERMINAL = ('complete', 'content_failed', 'transport_failed')
# Never judged: may be re-sent once the owner unblocks the ledger.
RESENDABLE = ('usage_limited', 'auth_failed')
_USAGE_LIMIT = ('usage limit', 'rate limit', 'limit reached', 'limit will reset', 'too many requests')
_AUTH = ('not logged in', '/login', 'invalid api key', 'authentication')
_PROBE_SCHEMA = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']}


def _save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


class CallLedger:
    """Durable per-job call accounting. A job with a terminal entry is never re-sent;
    a usage-limited or auth-failed job (never judged) is re-sendable after unblock()."""

    def __init__(self, path, cap=CALL_CAP):
        self.path, self.cap = Path(path), cap
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {'entries': {}, 'cap': self.cap}

    def _locked(self, fn):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self._load()
            result = fn(state)
            _save(self.path, state)
            return result

    def read(self):
        return self._load()

    @staticmethod
    def _used(state):
        return sum(e.get('attempts', 1) + sum(h.get('attempts', 1) for h in e.get('history', []))
                   for e in state['entries'].values())

    def calls_used(self):
        return self._used(self._load())

    def done(self, job_id):
        return self._load()['entries'].get(job_id, {}).get('status') in TERMINAL

    def block(self, reason):
        self._locked(lambda s: s.update(blocked_reason=reason))

    def unblock(self):
        self._locked(lambda s: s.pop('blocked_reason', None))

    def reserve(self, job_id):
        def go(state):
            if state.get('blocked_reason'):
                raise ValueError('ledger blocked: ' + state['blocked_reason'])
            entry = state['entries'].get(job_id)
            history = []
            if entry is not None:
                if entry['status'] not in RESENDABLE:
                    raise ValueError(f'{job_id} already has a {entry["status"]} entry; never re-sent automatically')
                history = entry.pop('history', []) + [entry]
            if self._used(state) >= self.cap:
                raise ValueError(f'call cap {self.cap} reached')
            state['entries'][job_id] = {'status': 'reserved', 'reserved_at': time.time(), 'history': history}
        self._locked(go)

    def settle(self, job_id, status, usage, attempts=1):
        def go(state):
            entry = state['entries'][job_id]
            if entry['status'] != 'reserved':
                raise ValueError('reservation already settled')
            entry.update(status=status, usage=usage, attempts=attempts, finished_at=time.time())
        self._locked(go)


def build_command(exe, system_prompt, schema, output_format='json'):
    cmd = [exe, '-p', '--model', MODEL, '--output-format', output_format, '--json-schema', json.dumps(schema),
           '--tools', '', '--strict-mcp-config', '--disable-slash-commands', '--no-session-persistence',
           '--setting-sources', 'project', '--system-prompt', system_prompt]
    return cmd + (['--verbose'] if output_format == 'stream-json' else [])


def cli_version(exe):
    return subprocess.run([exe, '--version'], capture_output=True, text=True, timeout=30).stdout.strip()


def _run(cmd, stdin, timeout):
    """One subprocess attempt in a fresh empty directory that is removed afterwards."""
    work = tempfile.mkdtemp(prefix='judge-')
    try:
        proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout, cwd=work)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return None, '', 'timeout'
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _envelope(stdout):
    for line in reversed(stdout.strip().splitlines()):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get('type') == 'result':
            return data
    return None


def _classify(code, stdout, stderr):
    """-> (status, raw, usage, envelope). 'transport' means retryable."""
    env = _envelope(stdout)
    if env is None:
        return 'transport', None, None, None
    text = str(env.get('result', '')).lower()
    if env.get('is_error'):
        if any(s in text for s in _AUTH):
            return 'auth_failed', None, None, env
        if env.get('api_error_status') == 429 or any(s in text for s in _USAGE_LIMIT):
            return 'usage_limited', None, None, env
        status = env.get('api_error_status')
        return ('transport' if status is None or status >= 500 else 'content_failed'), None, None, env
    raw = env.get('structured_output')
    if raw is None:
        try:
            raw = json.loads(env.get('result', ''))
        except ValueError:
            return 'content_failed', None, env.get('usage'), env
    return 'complete', raw, env.get('usage'), env


def call(payload, *, exe, ledger, job_id, timeout=300, system_prompt=None, schema=None):
    from .judge_v3 import OUTPUT_SCHEMA, rubric_text
    system_prompt = rubric_text() if system_prompt is None else system_prompt
    schema = OUTPUT_SCHEMA if schema is None else schema
    ledger.reserve(job_id)
    cmd = build_command(exe, system_prompt, schema)
    stdin = 'TASK:\n' + json.dumps(payload, ensure_ascii=False)
    attempts, status, raw, usage, env = 0, 'transport', None, None, None
    while attempts < 2 and status == 'transport':
        attempts += 1
        status, raw, usage, env = _classify(*_run(cmd, stdin, timeout))
    if status == 'transport':
        status = 'transport_failed'
    if status in ('usage_limited', 'auth_failed'):
        ledger.settle(job_id, status, usage, attempts)
        ledger.block(f'{status}: ' + str((env or {}).get('result', ''))[:200])
    else:
        ledger.settle(job_id, status, usage, attempts)
    return {'status': status, 'raw': raw, 'usage': usage, 'attempts': attempts,
            'cost_usd_reported': (env or {}).get('total_cost_usd'), 'cli_version': cli_version(exe)}


def isolation_probe(exe, timeout=120):
    """Canary call; fails if anything beyond the bare model reaches the judge's context."""
    cmd = build_command(exe, 'Return ok=true.', _PROBE_SCHEMA, output_format='stream-json')
    code, stdout, stderr = _run(cmd, 'Return ok=true.', timeout)
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    failures = []
    init = next((e for e in events if e.get('type') == 'system' and e.get('subtype') == 'init'), None)
    if init is None:
        failures.append('no init event')
    else:
        # --json-schema is delivered through Claude Code's internal StructuredOutput tool;
        # it is the only tool allowed in the judge's context.
        tools = [t for t in init.get('tools', []) if t not in ALLOWED_TOOLS]
        if tools:
            failures.append(f'unexpected tools: {tools}')
        for field in ('mcp_servers', 'slash_commands', 'skills'):
            if init.get(field):
                failures.append(f'{field} not empty: {init[field]}')
        extra = [p.get('name') for p in init.get('plugins', []) if p.get('path') != 'builtin']
        if extra:
            failures.append(f'non-builtin plugins loaded: {extra}')
    hooks = [e for e in events if 'hook' in str(e.get('subtype', '')).lower() or 'hook' in str(e.get('type', '')).lower()]
    if hooks:
        failures.append(f'{len(hooks)} hook events in stream')
    env = _envelope(stdout)
    if env is not None and env.get('is_error'):
        failures.append('canary call failed: ' + str(env.get('result', ''))[:120])
    from .io import digest
    return {'passed': not failures, 'failures': failures, 'cli_version': cli_version(exe),
            'init_event_hash': digest(json.dumps({k: v for k, v in (init or {}).items()
                                                  if k not in ('session_id', 'uuid', 'cwd', 'original_cwd',
                                                               'memory_paths', 'messaging_socket_path',
                                                               'startup_timing')}, sort_keys=True)),
            'checked_at': time.time()}


def require_run_preconditions(manifest, exe):
    if manifest.get('terms_confirmed_by_owner') is not True:
        raise ValueError('owner has not confirmed subscription terms for batch evaluation (spec §6.5)')
    probe = manifest.get('probe') or {}
    if not probe.get('passed'):
        raise ValueError('isolation probe has not passed')
    if probe.get('cli_version') != cli_version(exe):
        raise ValueError('Claude Code version changed since the isolation probe; re-run the probe')
