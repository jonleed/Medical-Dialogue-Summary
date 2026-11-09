"""Canonical section keys for grouping/reporting.

`requested_section` builds the Flask prompt, so it is never rewritten; the
canonical key is a separate field used only for aggregation.
"""


def canonical_section(case):
    return ' '.join(str(case.get('requested_section') or '').lower().split())


def with_canonical_sections(cases):
    return [dict(c, canonical_section=canonical_section(c)) for c in cases]
