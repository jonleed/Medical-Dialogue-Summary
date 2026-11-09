#!/usr/bin/env python3
"""Fake `claude` CLI for offline backend tests. Behavior chosen by FAKE_CLAUDE_MODE.

Envelopes mirror real Claude Code 2.1.311 output captured in claude_not_logged_in.json
and claude_init_event.json (same directory).
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
mode = os.environ.get('FAKE_CLAUDE_MODE', 'success')
state = Path(os.environ.get('FAKE_CLAUDE_STATE', '/dev/null'))
args = sys.argv[1:]

if args == ['--version']:
    print(os.environ.get('FAKE_CLAUDE_VERSION', '2.1.311 (Claude Code)'))
    sys.exit(0)

stdin = sys.stdin.read()
if os.environ.get('FAKE_CLAUDE_STDIN_LOG'):
    with open(os.environ['FAKE_CLAUDE_STDIN_LOG'], 'a') as f:
        f.write(json.dumps({'args': args, 'stdin': stdin, 'cwd': os.getcwd()}) + '\n')

calls = 0
if state.name != 'null':
    calls = int(state.read_text()) if state.exists() else 0
    state.write_text(str(calls + 1))

base = json.loads((HERE / 'claude_not_logged_in.json').read_text())
usage = {'input_tokens': 900, 'output_tokens': 120, 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}


def envelope(**kw):
    e = dict(base, is_error=False, terminal_reason='completed', usage=usage, total_cost_usd=0.01, result='')
    e.update(kw)
    return e


STRUCTURED = json.loads(os.environ.get('FAKE_CLAUDE_OUTPUT', '{"ok": true}'))

if mode.startswith('probe'):
    init = json.loads((HERE / 'claude_init_event.json').read_text())
    if mode == 'probe_user_plugin':
        init['plugins'] = init['plugins'] + [{'name': 'superpowers', 'path': '/Users/x/.claude/plugins/superpowers', 'source': 'superpowers@market'}]
    if mode in ('probe_clean', 'probe_hook'):
        init['tools'] = ['StructuredOutput']  # real 2.1.311 behaviour with --json-schema
    if mode == 'probe_tools':
        init['tools'] = ['StructuredOutput', 'Bash']
    print(json.dumps(init))
    if mode == 'probe_hook':
        print(json.dumps({'type': 'system', 'subtype': 'hook_response', 'hook_name': 'SessionStart', 'output': 'You have superpowers.'}))
    print(json.dumps(envelope(structured_output={'ok': True})))
elif mode == 'success':
    print(json.dumps(envelope(structured_output=STRUCTURED)))
elif mode == 'result_text':
    print(json.dumps(envelope(result=json.dumps(STRUCTURED))))
elif mode == 'invalid_json':
    print(json.dumps(envelope(result='I think the summary is fine.')))
elif mode == 'not_logged_in':
    print(json.dumps(base))
    sys.exit(1)
elif mode == 'usage_limit':
    print(json.dumps(dict(base, result='Claude usage limit reached. Your limit will reset at 5pm.', terminal_reason='api_error', api_error_status=429)))
    sys.exit(1)
elif mode == 'usage_limit_soft':  # limit text in a non-error result envelope
    print(json.dumps(envelope(result='Claude usage limit reached. Your limit will reset at 5pm.')))
elif mode == 'timeout':
    time.sleep(5)
elif mode == 'timeout_then_success':
    if calls == 0:
        time.sleep(5)
    print(json.dumps(envelope(structured_output=STRUCTURED)))
elif mode == 'crash_then_success':
    if calls == 0:
        print('Segmentation fault', file=sys.stderr)
        sys.exit(139)
    print(json.dumps(envelope(structured_output=STRUCTURED)))
