"""Output-validity gate (spec §3.2). Only 'summary' outputs enter factuality denominators."""
import re
from collections import Counter

ECHO_PREFIX = 'Summarize the following doctor-patient dialogue'
ECHO_8GRAM_SHARE = 0.6
DEGENERATE_4GRAM_SHARE = 0.3
# Ratio rules are undefined on short outputs (one n-gram can be 100% of them);
# see results/factuality/local/v3/validity_threshold_log.md.
MIN_WORDS_FOR_RATIO = 20
LABELS = ('summary', 'prompt_echo', 'empty', 'degenerate')
_LABEL_PREFIX = re.compile(r'^\s*[a-z/ ]{2,40}:\s*', re.I)
_WORD = re.compile(r'\w+')


def _words(text):
    return _WORD.findall(text.lower())


def _ngrams(words, n):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def classify_output(prediction, dialogue):
    text = prediction or ''
    if ECHO_PREFIX.lower() in text.lower():
        return 'prompt_echo'
    body = _LABEL_PREFIX.sub('', text, count=1)
    if not body.strip():
        return 'empty'
    words = _words(body)
    if len(words) < MIN_WORDS_FOR_RATIO:
        return 'summary'
    grams8 = _ngrams(words, 8)
    if grams8:
        source = set(_ngrams(_words(dialogue or ''), 8))
        if sum(g in source for g in grams8) / len(grams8) >= ECHO_8GRAM_SHARE:
            return 'prompt_echo'
    grams4 = _ngrams(words, 4)
    if grams4 and Counter(grams4).most_common(1)[0][1] / len(grams4) >= DEGENERATE_4GRAM_SHARE:
        return 'degenerate'
    return 'summary'


def validity_summary(cases):
    per_case = {c['case_id']: classify_output(c.get('prediction'), c.get('dialogue', '')) for c in cases}
    counts = {label: sum(v == label for v in per_case.values()) for label in LABELS}
    return {'counts': counts, 'rate_summary': counts['summary'] / len(cases) if cases else None,
            'per_case': per_case}
