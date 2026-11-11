#!/usr/bin/env python3
"""Toy external summarizer for generator tests: echoes the first 40 characters of each dialogue."""
import json
import os
import sys

rows = [json.loads(line) for line in sys.stdin if line.strip()]
if os.environ.get('ECHO_DROP_LAST'):
    rows = rows[:-1]
for r in rows:
    print(json.dumps({'case_id': r['case_id'], 'prediction': r['dialogue'][:40]}))
