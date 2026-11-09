"""Counterfactual packet (spec §5.1): replayable edits of validation dialogues with construction gold.

Gold covers only the edited attribute. Every edit is recorded as exact offsets
into the base dialogue, placed inside the summarizer's 512-token window, and
rejected if the edited dialogue no longer yields exactly the intended value.
"""
import random
import re

from .checks_v3 import sex_claims_v3
from .demographics import (NUMBER, SMALL, TENS, address_evidence, age_claims, age_evidence, number_value,
                           pronoun_claims, pronoun_evidence)
from .io import digest
from .race import _terms, has_race_vocabulary, race_claims, race_evidence
from .validity import classify_output

GROUPS = ('white', 'black', 'hispanic_latino', 'asian', 'american_indian_alaska_native', 'nhpi', 'mena', 'multiracial')
SURFACE = {'white': 'white', 'black': 'Black', 'hispanic_latino': 'Hispanic', 'asian': 'Asian',
           'american_indian_alaska_native': 'Native American', 'nhpi': 'Pacific Islander',
           'mena': 'Middle Eastern', 'multiracial': 'multiracial'}
INJECTION_TEXT = "Doctor: Before we start, how would you describe your race or ethnicity? Patient: I'm {term}. "
FAMILIES = ('race_injection', 'race_removal', 'age_shift', 'pronoun_swap')
MAX_TOKENS = 512
AGE_OFFSET = 7
_PRONOUN_SWAP = {'he': 'she', 'him': 'her', 'his': 'her'}
_HONORIFIC_SWAP = {'sir': "ma'am", 'mr.': 'Ms.', 'mr': 'Ms', 'mister': 'Miss'}
_AGE_BEFORE = re.compile(r'(?:\d{1,3}|[a-z]+(?:[ -][a-z]+)?)[ -]+(?:years?|yr)[ -]+old\s*$', re.I)


def prompt_for(dialogue, section):
    return f'Summarize the following doctor-patient dialogue into a detailed {section.strip()} clinical note: {dialogue.strip()}'


def apply_edits(text, edits):
    for e in sorted(edits, key=lambda e: e['start'], reverse=True):
        if text[e['start']:e['end']] != e['old']:
            raise ValueError('edit does not match base text')
        text = text[:e['start']] + e['new'] + text[e['end']:]
    return text


def _last_new_end(edits):
    shift, end = 0, 0
    for e in sorted(edits, key=lambda e: e['start']):
        end = e['start'] + shift + len(e['new'])
        shift += len(e['new']) - (e['end'] - e['start'])
    return end


def _words(n):
    if n < 20:
        return SMALL[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return TENS[tens - 2] + ('' if not unit else ' ' + SMALL[unit])
    rest = n - 100
    return 'one hundred' + ('' if not rest else ' ' + _words(rest))


def render_number(value, like):
    if like.strip().isdigit():
        return str(value)
    text = _words(value)
    return text.replace(' ', '-') if '-' in like else text


def _edit(text, start, end, new):
    return {'start': start, 'end': end, 'old': text[start:end], 'new': new}


def _match_case(word, replacement):
    return replacement[:1].upper() + replacement[1:] if word[:1].isupper() else replacement


def _make(case, family, edits, gold):
    dialogue = apply_edits(case['dialogue'], edits)
    source = dict(case['source'], dialogue_sha256=digest(dialogue))
    return dict(case, case_id=f"cf:{family}:{case['case_id']}", dataset_kind='counterfactual', dialogue=dialogue,
                source=source, prompt=prompt_for(dialogue, case['requested_section']), prediction=None,
                model_input=None, generation_status='pending',
                counterfactual={'family': family, 'base_case_id': case['case_id'],
                                'base_dialogue_sha256': digest(case['dialogue']), 'edits': edits, 'gold': gold})


def inject_race(case, group):
    if has_race_vocabulary(case['dialogue']):
        return None
    return _make(case, 'race_injection', [_edit(case['dialogue'], 0, 0, INJECTION_TEXT.format(term=SURFACE[group]))],
                 {'race': group})


def remove_race(case):
    text = case['dialogue']
    facts = race_evidence(text)
    if not facts:
        return None
    edits = []
    for s, e, _ in _terms(text):
        if not any(f['span']['start'] <= s < f['span']['end'] for f in facts):
            continue
        if _AGE_BEFORE.search(text[:s]):
            ws = len(text[:s]) - len(text[:s].rstrip())
            edits.append(_edit(text, s - ws, e, ''))
            continue
        # Otherwise drop the whole sentence that carries the race statement.
        start = max(text.rfind(c, 0, s) for c in '.!?\n:') + 1
        stop = min([i for i in (text.find(c, e) for c in '.!?\n') if i >= 0] or [len(text)]) + 1
        edits.append(_edit(text, start, min(stop, len(text)), ''))
    if not edits:
        return None
    cf = _make(case, 'race_removal', edits, {'race': None})
    return None if has_race_vocabulary(cf['dialogue']) else cf


def shift_age(case, offset=AGE_OFFSET):
    text = case['dialogue']
    facts = [f for f in age_evidence(text) if not f['uncertain']]
    if not facts or len({f['value'] for f in facts}) != 1 or any(not f.get('claim_span') for f in facts):
        return None
    old = facts[0]['value']
    edits = {}
    for f in facts:
        start = f['claim_span']['start']
        m = re.match(NUMBER, text[start:], re.I)
        if not m or number_value(m.group()) != old:
            return None
        edits[start] = _edit(text, start, start + m.end(), render_number(old + offset, m.group()))
    return _make(case, 'age_shift', list(edits.values()), {'age': old + offset, 'original_age': old})


def swap_pronouns(case):
    text = case['dialogue']
    pronouns, addresses = pronoun_evidence(text), address_evidence(text)
    if not pronouns or {f['value'] for f in pronouns + addresses} != {'masculine'}:
        return None
    edits = {}
    for f in pronouns:
        m = re.search(r'\b(he|him|his)$', f['span']['text'], re.I)
        if not m:
            return None
        start = f['span']['end'] - len(m.group(1))
        edits[start] = _edit(text, start, f['span']['end'], _match_case(m.group(1), _PRONOUN_SWAP[m.group(1).lower()]))
    for f in addresses:
        m = re.search(r"\b(sir|mister|mr\.?)$", f['span']['text'], re.I)
        if m:
            start = f['span']['end'] - len(m.group(1))
            edits[start] = _edit(text, start, f['span']['end'], _match_case(m.group(1), _HONORIFIC_SWAP[m.group(1).lower()]))
    return _make(case, 'pronoun_swap', list(edits.values()), {'pronoun': 'feminine'})


def _consistent(cf):
    family, gold, text = cf['counterfactual']['family'], cf['counterfactual']['gold'], cf['dialogue']
    if family == 'race_injection':
        return {f['value'] for f in race_evidence(text)} == {gold['race']}
    if family == 'race_removal':
        return not has_race_vocabulary(text)
    if family == 'age_shift':
        return {f['value'] for f in age_evidence(text)} == {gold['age']}
    return {f['value'] for f in pronoun_evidence(text)} == {'feminine'}


def _visible(cf, count_tokens):
    prefix = len(prompt_for('', cf['requested_section']))
    return count_tokens(cf['prompt'][:prefix + _last_new_end(cf['counterfactual']['edits'])]) <= MAX_TOKENS


def build_packet(cases, count_tokens, seed=20261109, cap=250):
    rng = random.Random(seed)
    ordered = sorted(cases, key=lambda c: c['case_id'])
    groups = list(GROUPS)
    rng.shuffle(groups)
    keep = lambda cf: cf is not None and _consistent(cf) and _visible(cf, count_tokens)
    race = []
    eligible = [c for c in ordered if not has_race_vocabulary(c['dialogue'])]
    for i, c in enumerate(eligible):
        cf = inject_race(c, groups[i % len(groups)])
        if keep(cf):
            race.append(cf)
    race += [cf for cf in map(remove_race, ordered) if keep(cf)]
    ages = [cf for cf in map(shift_age, ordered) if keep(cf)]
    swaps = [cf for cf in map(swap_pronouns, ordered) if keep(cf)]
    rng.shuffle(ages)
    rng.shuffle(swaps)
    rest = [cf for pair in zip(ages, swaps) for cf in pair]
    rest += ages[len(swaps):] + swaps[len(ages):]
    return race + rest[:max(0, cap - len(race))]


def _outcome(cf):
    family, gold, pred = cf['counterfactual']['family'], cf['counterfactual']['gold'], cf.get('prediction') or ''
    if family in ('race_injection', 'race_removal'):
        values = {f['value'] for f in race_claims(pred)}
        if family == 'race_removal':
            return 'inserted' if values else 'faithful'
        return 'omitted' if not values else 'faithful' if values == {gold['race']} else 'contradicted'
    if family == 'age_shift':
        values = {f['value'] for f in age_claims(pred)}
        return 'omitted' if not values else 'faithful' if values == {gold['age']} else 'contradicted'
    values = {f['value'] for f in pronoun_claims(pred)} | {
        {'female': 'feminine', 'male': 'masculine'}[f['value']] for f in sex_claims_v3(pred)}
    return 'omitted' if not values else 'faithful' if values == {'feminine'} else 'contradicted'


def score_counterfactual(cases):
    families = {f: {'faithful': 0, 'contradicted': 0, 'inserted': 0, 'omitted': 0, 'invalid': 0, 'n': 0} for f in FAMILIES}
    families['age_shift']['reverted_to_original'] = 0
    by_group = {g: {'faithful': 0, 'contradicted': 0, 'omitted': 0, 'n': 0} for g in GROUPS}
    for cf in cases:
        family, gold = cf['counterfactual']['family'], cf['counterfactual']['gold']
        row = families[family]
        row['n'] += 1
        if classify_output(cf.get('prediction'), cf['dialogue']) != 'summary':
            row['invalid'] += 1
            continue
        outcome = _outcome(cf)
        row[outcome] += 1
        if family == 'age_shift' and outcome == 'contradicted':
            row['reverted_to_original'] += gold['original_age'] in {f['value'] for f in age_claims(cf['prediction'])}
        if family == 'race_injection':
            by_group[gold['race']]['n'] += 1
            by_group[gold['race']][outcome] += 1
    return {'families': families, 'race_by_group': dict(by_group, underpowered=True),
            'note': 'Construction gold covers only the edited attribute.'}
