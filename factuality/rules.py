"""Rule checker v3. v2 (`checks.py`) stays frozen; this module only wraps it.

Adds race/ethnicity (with an `unsupported` label for race absent from the whole
source) and parses sex claims with descriptors between age and sex. Sex terms and
pronouns are one gender-reference dimension, checked against every cue (see
`checks.gender_reference_suggestion`), so there is no separate cross-dimension flag.
"""
import re

from .checks import _suggest, automatic_summary, gender_reference_suggestion
from .demographics import (AGE_UNIT, NUMBER, _fact, age_claims, age_evidence, identity_claims, identity_evidence,
                           pronoun_claims)
from .race import has_race_vocabulary, proxy_cues, race_claims, race_evidence

CHECKER_VERSION = 'patient_demographics_v3.1'
_SEX_CLAIM = re.compile(r'\b(?:the patient is|this is)\s+(?:an?\s+)?(?:' + NUMBER + AGE_UNIT + r'\s+)?'
                        r'(?:[\w-]+\s+){0,3}?(female|male)\b(?![-/]|\s+to\b)', re.I)


def sex_claims_v3(text):
    return [_fact(text, m.group(1).lower(), m.start(1), m.end(1), 'summary_patient_sex') for m in _SEX_CLAIM.finditer(text)]


def _race(case, claim, source, visible):
    facts, visible_facts = race_evidence(source), race_evidence(visible)
    s = _suggest(case, 'race_ethnicity', claim, facts, visible_facts)
    if s['proposed_label'] != 'unclear' or s['issue'] == 'conflicting_or_uncertain_source':
        return s
    if not has_race_vocabulary(source) and not proxy_cues(source):
        s.update(proposed_label='unsupported', issue='race_inserted', review_required=True,
                 model_visible_proposed_label='unsupported',
                 rationale='No race or ethnicity term or proxy cue appears anywhere in the full source; '
                           'the generated race is a candidate insertion for review.')
    else:
        s.update(issue='race_not_established',
                 rationale='Race vocabulary or proxy cues appear in the source but no explicit patient race was '
                           'established. Names, language and nationality are not race evidence.')
    return s


def suggestions_v3(case):
    source = case['dialogue']
    output = case.get('prediction') or ''
    visible = (case.get('model_input') or {}).get('text', '')
    out = []
    for typ, extract, claims in [('age', age_evidence, age_claims),
                                 ('gender', lambda t: identity_evidence(t, 'gender'), lambda t: identity_claims(t, 'gender'))]:
        for claim in claims(output):
            out.append(_suggest(case, typ, claim, extract(source), extract(visible)))
    out += [gender_reference_suggestion(case, 'pronoun', c) for c in pronoun_claims(output)]
    out += [gender_reference_suggestion(case, 'sex', c) for c in sex_claims_v3(output)]
    out += [_race(case, claim, source, visible) for claim in race_claims(output)]
    for s in out:
        s.update(method=CHECKER_VERSION, subject='patient')
    return out


def automatic_summary_v3(suggestions):
    summary = automatic_summary(suggestions)
    summary.update(checker_version=CHECKER_VERSION,
                   unsupported_suggestions=sum(s['proposed_label'] == 'unsupported' for s in suggestions),
                   race_inserted=sum(s['issue'] == 'race_inserted' for s in suggestions))
    return summary
