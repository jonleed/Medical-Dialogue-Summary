"""Explicit race/ethnicity extraction (spec §4.1). Proposals only, never clinical truth.

Race is taken only from explicit statements about the patient. Names, language,
nationality, country of origin and location are never race evidence; they are
returned separately as proxy cues so a reviewer can see them.
"""
import re

from .demographics import OTHER_PERSON, _fact, _unique, qualified_context, turns

RACE_TERMS = {
    'white': 'white', 'caucasian': 'white',
    'black': 'black', 'african american': 'black', 'african-american': 'black',
    'hispanic': 'hispanic_latino', 'latino': 'hispanic_latino', 'latina': 'hispanic_latino',
    'latinx': 'hispanic_latino',
    'asian': 'asian', 'asian american': 'asian', 'chinese american': 'asian',
    'korean american': 'asian', 'vietnamese american': 'asian', 'japanese american': 'asian',
    'filipino': 'asian', 'filipina': 'asian', 'indian american': 'asian',
    'american indian': 'american_indian_alaska_native', 'native american': 'american_indian_alaska_native',
    'alaska native': 'american_indian_alaska_native',
    'native hawaiian': 'nhpi', 'pacific islander': 'nhpi',
    'middle eastern': 'mena', 'north african': 'mena', 'arab american': 'mena',
    'multiracial': 'multiracial', 'biracial': 'multiracial', 'mixed race': 'multiracial',
}
_TERM = re.compile(r'\b(' + '|'.join(re.escape(t).replace(r'\ ', r'\s+')
                                     for t in sorted(RACE_TERMS, key=len, reverse=True)) + r')\b', re.I)
_EXCLUDE = re.compile(r'\b(?:white\s+blood\s+cells?|white\s+(?:cell\s+)?counts?|white\s+matter|'
                      r'black\s+(?:stools?|tarry|eyes?|and\s+blue|mold)|black[- ]?outs?)\b', re.I)
_AGE = r'(?:\d{1,3}|[a-z]+(?:[ -][a-z]+)?)[ -]+(?:years?|yr)[ -]+old'
_BEFORE = re.compile(r'(?:(?:the\s+)?patient\s+is|this\s+is|(?:he|she)\s+is)\s+(?:an?\s+)?(?:' + _AGE + r'\s+)?$'
                     r'|' + _AGE + r'\s+$', re.I)
_AFTER = re.compile(r'\s+(?:male|female|man|woman|gentleman|lady|boy|girl|patient)\b', re.I)
_SELF = re.compile(r"\bI(?:'m|\s+am|\s+identify\s+as)\s+(?:an?\s+)?(?:" + _AGE + r"\s+)?$", re.I)
_CONFIRM_Q = re.compile(r"\byou\s+(?:identify\s+as|are)\s+(?:an?\s+)?$", re.I)
_YES = re.compile(r"\s*(?:yes|yeah|correct|that(?:'s|\s+is)\s+(?:right|correct))\b", re.I)
_RACE_QUESTION = re.compile(r'\b(?:race|ethnicity|ethnic\s+background|how\s+(?:do|would)\s+you\s+'
                            r'(?:identify|describe\s+(?:yourself|your\s+(?:race|ethnicity))))\b', re.I)
_PROXY = re.compile(r"\bI(?:'m|\s+am)\s+from\s+[A-Z][\w-]+|\bI\s+speak\s+[A-Z][\w-]+|\bborn\s+in\s+[A-Z][\w-]+"
                    r"|\bmy\s+(?:first\s+|native\s+)?language\s+is\s+[A-Za-z][\w-]+")


def _norm(text):
    return text.replace('’', "'")  # same length, so offsets stay valid


def _terms(text):
    t = _norm(text)
    excluded = [m.span() for m in _EXCLUDE.finditer(t)]
    out = []
    for m in _TERM.finditer(t):
        if any(s <= m.start() < e for s, e in excluded):
            continue
        out.append((m.start(), m.end(), RACE_TERMS[' '.join(m.group(1).lower().split())]))
    return out


def has_race_vocabulary(text):
    return bool(_terms(text))


def _sentences(text):
    return [(m.start(), m.group()) for m in re.finditer(r'[^.!?\n]+[.!?]?', text)]


def _anchored(sentence, start, end):
    """Patient-introduction construction around a race term within one sentence."""
    before = _norm(sentence[:start])
    if OTHER_PERSON.search(before):
        return False
    return bool(_BEFORE.search(before) or _AFTER.match(sentence[end:]))


def race_claims(text):
    result = []
    for base, sentence in _sentences(text):
        for s, e, value in _terms(sentence):
            if _anchored(sentence, s, e):
                result.append(_fact(text, value, base + s, base + e, 'summary_patient_race'))
    return _unique(result)


def race_evidence(text):
    dialogue = turns(text)
    if not dialogue:
        return [dict(f, basis='explicit_patient_race') for f in race_claims(text)]
    result = []
    for i, turn in enumerate(dialogue):
        body, role = turn['body'], turn['role']
        previous = dialogue[i - 1] if i else None
        for s, e, value in _terms(body):
            before = _norm(body[:s])
            uncertain = qualified_context(body, _TERM.search(_norm(body), s))
            if role == 'patient':
                sentence_before = re.split(r'[.!?\n]', before)[-1]
                if OTHER_PERSON.search(sentence_before):
                    continue
                if _SELF.search(sentence_before):
                    result.append(_fact(text, value, turn['start'], turn['body_start'] + e, 'explicit_self_race', uncertain))
                elif (previous and previous['role'] == 'doctor' and _RACE_QUESTION.search(previous['body'])
                      and not sentence_before.strip(" ,'")):
                    result.append(_fact(text, value, previous['start'], turn['body_start'] + e, 'race_question_answer', uncertain))
            elif role in {'doctor', 'guest_clinician'}:
                nxt = dialogue[i + 1] if i + 1 < len(dialogue) else None
                sentence_before = re.split(r'[.!?\n]', before)[-1]
                if _CONFIRM_Q.search(sentence_before):
                    if nxt and nxt['role'] == 'patient' and _YES.match(_norm(nxt['body'])):
                        result.append(_fact(text, value, turn['start'], nxt['end'], 'confirmed_race_question', uncertain))
                    continue
                sentence_start = max(before.rfind(c) for c in '.!?\n') + 1
                if _anchored(body[sentence_start:], s - sentence_start, e - sentence_start):
                    result.append(_fact(text, value, turn['start'], turn['body_start'] + e, 'explicit_patient_race', uncertain))
    return _unique(result)


def proxy_cues(text):
    t = _norm(text)
    return [{'span': {'start': m.start(), 'end': m.end(), 'text': text[m.start():m.end()]}, 'basis': 'proxy_cue'}
            for m in _PROXY.finditer(t)]
