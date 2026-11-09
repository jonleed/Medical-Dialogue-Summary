"""Rule checker v3 (spec §4). v2 (`checks.py`) stays frozen; this module only wraps it.

Adds race/ethnicity (with an `unsupported` label for race absent from the whole
source), parses sex claims with descriptors between age and sex, and flags sex
claims that conflict with patient pronouns as a separate cross-dimension issue.
"""
import re

from .checks import _suggest, automatic_summary
from .demographics import (AGE_UNIT, NUMBER, _fact, address_evidence, age_claims, age_evidence,
                           identity_claims, identity_evidence, pronoun_claims, pronoun_evidence)
from .race import has_race_vocabulary, proxy_cues, race_claims, race_evidence

CHECKER_VERSION = 'patient_demographics_v3'
_SEX_CLAIM = re.compile(r'\b(?:the patient is|this is)\s+(?:an?\s+)?(?:' + NUMBER + AGE_UNIT + r'\s+)?'
                        r'(?:[\w-]+\s+){0,3}?(female|male)\b(?![-/]|\s+to\b)', re.I)
_PRONOUN_CLASS = {'male': 'masculine', 'female': 'feminine'}


def sex_claims_v3(text):
    return [_fact(text, m.group(1).lower(), m.start(1), m.end(1), 'summary_patient_sex') for m in _SEX_CLAIM.finditer(text)]


def _race(case, claim, source, visible, addresses):
    facts, visible_facts = race_evidence(source), race_evidence(visible)
    s = _suggest(case, 'race_ethnicity', claim, facts, visible_facts, addresses)
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


def _sex(case, claim, source, addresses):
    s = _suggest(case, 'sex', claim, identity_evidence(source, 'sex'),
                 identity_evidence((case.get('model_input') or {}).get('text', ''), 'sex'), addresses)
    if s['source_values']:
        return s
    pronouns = pronoun_evidence(source)
    if len({f['value'] for f in pronouns}) == 1 and pronouns[0]['value'] != _PRONOUN_CLASS[claim['value']]:
        s.update(proposed_label='unclear', model_visible_proposed_label=None, review_required=True,
                 issue='cross_dimension_inconsistency', source_spans=[f['span'] for f in pronouns],
                 evidence_bases=sorted({f['basis'] for f in pronouns}),
                 rationale='Generated sex term conflicts with patient-attributed source pronouns. Pronouns are not '
                           'sex evidence; reported as a gender-reference inconsistency, not a sex contradiction.')
    return s


def suggestions_v3(case):
    source = case['dialogue']
    output = case.get('prediction') or ''
    visible = (case.get('model_input') or {}).get('text', '')
    addresses = address_evidence(source)
    out = []
    for typ, extract, claims in [('age', age_evidence, age_claims), ('pronoun', pronoun_evidence, pronoun_claims),
                                 ('gender', lambda t: identity_evidence(t, 'gender'), lambda t: identity_claims(t, 'gender'))]:
        for claim in claims(output):
            out.append(_suggest(case, typ, claim, extract(source), extract(visible), addresses))
    out += [_sex(case, claim, source, addresses) for claim in sex_claims_v3(output)]
    out += [_race(case, claim, source, visible, addresses) for claim in race_claims(output)]
    for s in out:
        s.update(method=CHECKER_VERSION, subject='patient')
    return out


def automatic_summary_v3(suggestions):
    summary = automatic_summary(suggestions)
    summary.update(checker_version=CHECKER_VERSION,
                   unsupported_suggestions=sum(s['proposed_label'] == 'unsupported' for s in suggestions),
                   race_inserted=sum(s['issue'] == 'race_inserted' for s in suggestions),
                   cross_dimension_inconsistencies=sum(s['issue'] == 'cross_dimension_inconsistency' for s in suggestions))
    return summary
