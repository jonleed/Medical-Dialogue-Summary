"""Transparent patient-demographic extraction rules; no models or dataset IDs.

These rules establish candidates and evidence, not exhaustive clinical truth.
Ages require an age construction or a patient-directed age question. Pronouns
require a patient anchor/care context; patient turns about relatives/providers
never supply third-person patient identity. Honorifics are weaker address cues.
"""
import re

SMALL = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
TENS = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()
UNIT = '(?:' + '|'.join(SMALL[1:10]) + ')'
UNDER100 = '(?:' + '|'.join(TENS) + r')(?:[ -]+' + UNIT + ')?|(?:' + '|'.join(SMALL) + ')'
NUMBER = r'(?:\d{1,3}|one[ -]+hundred(?:[ -]+(?:and[ -]+)?(?:' + UNDER100 + '))?|(?:' + UNDER100 + '))'
AGE_UNIT = r'(?:[ -]+years?[ -]+old|[ -]+years?[ -]+of[ -]+age)'
SELF = r"\bI(?:\s+am|'m|’m)\s+(?P<approx>about\s+|around\s+|approximately\s+)?"
SELF_AGE = re.compile(SELF + r'(?P<number>' + NUMBER + r')(?P<unit>' + AGE_UNIT + r')\b', re.I)
SELF_BARE = re.compile(SELF + r'(?P<number>' + NUMBER + r')(?=\s*(?:[.,!?;]|$|right now\b))', re.I)
TURNED = re.compile(r"\bI\s+(?:have\s+)?(?:just\s+)?turned\s+(?P<number>" + NUMBER + r')(?=\s*(?:[.,!?;]|$|today\b))', re.I)
SELF_DESCRIPTION = re.compile(r'\b(?:a|an)\s+(?P<number>' + NUMBER + r')' + AGE_UNIT + r'\s+(?:woman|man|person|patient)\s+like\s+me\b', re.I)
PATIENT_AGE = re.compile(r'\b(?:the patient is|this is|this patient is)\s+(?:an?\s+)?(?P<number>' + NUMBER + r')' + AGE_UNIT + r'\b', re.I)
CONFIRM_AGE = re.compile(r'\byou\s+are\s+(?P<number>' + NUMBER + r')' + AGE_UNIT + r'\s*\?', re.I)
THIRD_AGE = re.compile(r'\b(?:he|she|the patient)\s+is\s+(?P<number>' + NUMBER + r')(?=' + AGE_UNIT + r'|\s*[.!?,;]|\s*$)', re.I)
AGE_QUESTION = re.compile(r'\b(?:how (?:old|young) are you|(?:tell me|get(?:ting)?|confirm).{0,25}your age|your age\s*\?)', re.I)
ROLE = re.compile(r'(?<!\w)(Doctor|Patient|Guest_family|Guest_clinician)\s*:', re.I)
PRONOUN = re.compile(r'\b(he|him|his|she|her|hers)\b', re.I)
OTHER_PERSON = re.compile(r"\b(?:mother|father|mom|dad|son|daughter|brother|sister|nephew|niece|wife|husband|partner|parents|friend|cousin|aunt|uncle|grandmother|grandfather|neighbor|doctor|physician|nurse|therapist|specialist|clinician|psychologist|paramedic|caregiver|another patient|psychiatrist|surgeon|assistant|case manager|corpsman)\b|\b(?:my|her|his|your|their)\s+doctor\b|\bDr\.", re.I)
CARE_CONTEXT = re.compile(r'\b(?:birth history|born|incubator|immunizations?|public school|special education|past medical history|medical history|surger\w*|review of (?:his|her) systems|(?:his|her) symptoms)\b', re.I)
HONORIFIC = re.compile(r"\b(sir|ma'am|ma’am|miss|mister|mr\.?|mrs\.?|ms\.?)\b", re.I)


def number_value(text):
    """Parse the finite English-cardinal grammar accepted by NUMBER."""
    if text.isdigit(): return int(text)
    values = {word:i for i,word in enumerate(SMALL)}
    values.update({word:(i+2)*10 for i,word in enumerate(TENS)})
    total = 0
    for word in re.split(r'[ -]+', text.lower()):
        if word == 'and': continue
        if word == 'hundred': total *= 100
        elif word in values: total += values[word]
        else: return None
    return total


def evidence(text, start, end):
    return {'start':start, 'end':end, 'text':text[start:end]}


def turns(text):
    matches = list(ROLE.finditer(text))
    return [{'role':m.group(1).lower(), 'start':m.start(), 'body_start':m.end(),
             'end':matches[i+1].start() if i+1<len(matches) else len(text),
             'body':text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)]}
            for i,m in enumerate(matches)]


def _fact(text, value, start, end, basis, uncertain=False, claim_span=None):
    while end > start and text[end-1].isspace(): end -= 1
    return {'value':value, 'span':evidence(text,start,end), 'basis':basis, 'uncertain':uncertain, 'claim_span':claim_span}


def _unique(facts):
    return list({(f['value'],f['span']['start'],f['span']['end']):f for f in facts}.values())


def qualified_context(body, match):
    """Conservatively abstain on quoted, uncertain, negated or historical clauses."""
    prefix=re.split(r'[.!?\n]',body[:match.start()])[-1]
    suffix=re.split(r'[.!?\n]',body[match.end():])[0]
    markers=r"\b(?:if|suppose|maybe|possibly|perhaps|think|thought|believe|not|never|said|says|told|used to|last year|years ago|old photo|back then|at that time)\b"
    return bool(re.search(markers,prefix,re.I) or re.search(r'\b(?:last year|years ago|old photo|back then|at that time)\b',suffix,re.I) or re.search(r"[\"‘“]|(?:^|[\s,])'",prefix))


def normalized_identity(word):
    return word.lower().replace('non-binary','nonbinary')


def age_evidence(text):
    """Source current ages attributed to the patient, with speaker/evidence spans."""
    result=[]; dialogue=turns(text)
    for i,turn in enumerate(dialogue):
        body=turn['body']; role=turn['role']; previous=dialogue[i-1] if i else None
        asked = bool(previous and previous['role']=='doctor' and AGE_QUESTION.search(previous['body']))
        if role=='patient':
            matches=[(m,'explicit_self_age',turn['start']) for m in SELF_AGE.finditer(body)]
            matches += [(m,'self_age_description',turn['start']) for m in SELF_DESCRIPTION.finditer(body)]
            matches += [(m,'birthday_self_age',turn['start']) for m in TURNED.finditer(body)]
            # Bare numbers are ages only in the immediately preceding age question,
            # or an explicit "I am ... right now" construction.
            for m in SELF_BARE.finditer(body):
                if asked or re.match(r'\s*right now\b',body[m.end():],re.I):
                    matches.append((m,'age_question_answer',previous['start'] if asked else turn['start']))
            for m,basis,start in matches:
                value=number_value(m.group('number'))
                if value is not None:
                    approx=bool(m.groupdict().get('approx'))
                    result.append(_fact(text,value,start,turn['body_start']+m.end(),basis,approx or value>120 or qualified_context(body,m), evidence(text,turn['body_start']+m.start('number'),turn['body_start']+m.end())))
            # A confirmed question supplies age evidence; an unconfirmed question does not.
            if previous and previous['role']=='doctor' and re.match(r'\s*(?:yes|correct|that is correct|that\x27s right)\b',body,re.I):
                m=CONFIRM_AGE.search(previous['body'])
                if m:
                    value=number_value(m.group('number'))
                    result.append(_fact(text,value,previous['start'],turn['end'],'confirmed_age_question',value>120))
        if role in {'doctor','guest_clinician'}:
            for m in PATIENT_AGE.finditer(body):
                # A question about a supposed age is not an established declaration.
                tail=body[m.end():].split('\n',1)[0]
                if '?' in tail.split('.',1)[0]: continue
                value=number_value(m.group('number'))
                result.append(_fact(text,value,turn['start'],turn['body_start']+m.end(),'explicit_patient_age',value>120 or qualified_context(body,m),evidence(text,turn['body_start']+m.start('number'),turn['body_start']+m.end())))
        if role=='guest_family' and asked:
            for m in THIRD_AGE.finditer(body):
                value=number_value(m.group('number'))
                result.append(_fact(text,value,previous['start'],turn['body_start']+m.end(),'family_answer_patient_age',value>120 or qualified_context(body,m),evidence(text,turn['body_start']+m.start('number'),turn['body_start']+m.end())))
    # Standalone source assertions (no speaker tags) retain the v1 use case.
    if not dialogue:
        for m in PATIENT_AGE.finditer(text):
            value=number_value(m.group('number'))
            result.append(_fact(text,value,m.start(),m.end(),'explicit_patient_age',value>120 or qualified_context(text,m)))
    return _unique(result)


def age_claims(text):
    result=[]
    for m in PATIENT_AGE.finditer(text):
        # Summary assertions are anchored to patient/intro, never relatives.
        start=m.start('number')
        result.append(_fact(text,number_value(m.group('number')),start,m.end(),'summary_patient_age'))
    # Dialogue-like model outputs may contain several distinct self assertions.
    if turns(text):
        for fact in age_evidence(text):
            fact = dict(fact)
            if fact.get('claim_span'): fact['span'] = fact['claim_span']
            result.append(fact)
    return _unique(result)


def pronoun_evidence(text):
    result=[]; dialogue=turns(text); patient_context=False; context_start=None
    if not dialogue:
        dialogue=[{'role':'doctor','start':0,'body_start':0,'end':len(text),'body':text}]
    for i,turn in enumerate(dialogue):
        body=turn['body']; role=turn['role']; previous=dialogue[i-1] if i else None
        if role=='patient':
            m=re.search(r'\b(?:my pronouns are|I use)\s+(she\s*/\s*her|he\s*/\s*him)\b',body,re.I)
            if m:
                value='feminine' if m.group(1).lower().startswith('she') else 'masculine'
                result.append(_fact(text,value,turn['start'],turn['body_start']+m.end(),'self_pronouns'))
            # Third-person pronouns in patient speech describe other people.
            patient_context=False
            continue
        anchor=re.search(r"\b(?:the patient(?!['’]s)|patients? case|your baby|your child|our little man)\b",body,re.I)
        explicit=bool(anchor)
        other=bool(OTHER_PERSON.search(body))
        care=bool(CARE_CONTEXT.search(body)) and not other
        nurse=re.search(r'\bNurse,?\s+(?:please\s+)?give\s+(him|her)\b',body,re.I)
        related=bool(previous and previous['role']=='doctor' and re.search(r'\b(?:related to the patient|relationship to the patient)\b',previous['body'],re.I))
        companion=None
        if role=='guest_family' and previous and re.search(r'\b(?:husband|wife|partner)\b',previous['body'],re.I):
            companion=re.search(r'\b(?:here|came|come|stay)\b.{0,35}\bwith (him|her)\b',body,re.I)
        # Explicit patient statements may contain other actors later. Limit their
        # evidence to pronouns before any newly introduced other-person noun.
        scope=explicit or care or nurse is not None or related or companion is not None or (patient_context and not other)
        matches=[]
        if companion: matches=[companion]
        elif nurse: matches=[nurse]
        elif scope:
            limit=OTHER_PERSON.search(body)
            safe=body[:limit.start()] if limit else body
            matches=list(PRONOUN.finditer(safe))
            if anchor:
                matches=[m for m in matches if m.start()>=anchor.end() or re.match(r'\s+is\s+the patient\b',body[m.end():],re.I)]
        evidence_start=turn['start']
        if related or companion: evidence_start=previous['start']
        elif patient_context and not (explicit or care or nurse): evidence_start=context_start
        for m in matches:
            word=m.group(1).lower()
            value='masculine' if word in {'he','him','his'} else 'feminine'
            # Full scoped utterance preserves the identity anchor; token alone
            # could be falsely matched to a surgeon/relative in truncated input.
            result.append(_fact(text,value,evidence_start,turn['body_start']+max(m.end(),anchor.end() if anchor else 0),'patient_pronoun_context',qualified_context(body,m)))
        patient_context=bool(scope and not other)
        context_start=evidence_start if patient_context else None
    return _unique(result)


def pronoun_claims(text):
    if turns(text): return pronoun_evidence(text)
    result=[]; patient=True
    # Clinical summary start defaults to the requested patient section. Explicit
    # relative/provider subjects switch scope until a new patient anchor appears.
    for m in re.finditer(r'[^.!?\n]+[.!?]?',text):
        sentence=m.group(); base=m.start()
        anchor=re.search(r'\bthe patient\b(?![\x27’]s)|\bthis is\s+(?:a|an)?\s*'+NUMBER+AGE_UNIT,sentence,re.I)
        other=OTHER_PERSON.search(sentence)
        if other and not anchor:
            patient=False
            continue
        if anchor: patient=True
        if patient:
            # Only references before a newly introduced third party are safe.
            safe=sentence[:other.start()] if other else sentence
            for p in PRONOUN.finditer(safe):
                if anchor and p.start()<anchor.end() and not re.match(r'\s+is\s+the patient\b',sentence[p.end():],re.I):continue
                word=p.group(1).lower()
                result.append(_fact(text,'masculine' if word in {'he','him','his'} else 'feminine',base+p.start(),base+p.end(),'summary_patient_pronoun'))
    # One first occurrence per pronoun category, avoiding repeated-token inflation.
    return list({f['value']:next(x for x in result if x['value']==f['value']) for f in result}.values())


def identity_evidence(text, dimension):
    result=[]
    if dimension=='sex':
        patterns=[re.compile(r'\b(?:the patient is|the patient\x27s sex is|patient sex is)\s+(?:a\s+)?(female|male)\b(?![-/]|\s+to\b)',re.I)]
    else:
        patterns=[re.compile(r'\bthe patient (?:identifies as|gender is)\s+(?:a\s+)?(woman|man|nonbinary|non-binary)\b(?![-/]|\s+to\b)',re.I)]
    for turn in turns(text):
        if turn['role']=='patient':
            pattern=re.compile(r'\bI am\s+(female|male)\b(?![-/]|\s+to\b)',re.I) if dimension=='sex' else re.compile(r'\bI (?:identify as|am)\s+(?:a\s+)?(woman|man|nonbinary|non-binary)\b(?![-/]|\s+to\b)',re.I)
            for m in pattern.finditer(turn['body']):
                result.append(_fact(text,normalized_identity(m.group(1)),turn['start'],turn['body_start']+m.end(),'explicit_self_'+dimension,qualified_context(turn['body'],m)))
        elif turn['role'] in {'doctor','guest_clinician'}:
            for pattern in patterns:
                for m in pattern.finditer(turn['body']):
                    result.append(_fact(text,normalized_identity(m.group(1)),turn['start'],turn['body_start']+m.end(),'explicit_patient_'+dimension,qualified_context(turn['body'],m)))
    if not turns(text):
        for pattern in patterns:
            for m in pattern.finditer(text):result.append(_fact(text,normalized_identity(m.group(1)),m.start(),m.end(),'explicit_patient_'+dimension,qualified_context(text,m)))
    return _unique(result)


def identity_claims(text, dimension):
    if dimension=='sex':
        pattern=re.compile(r'\b(?:the patient is|this is)\s+(?:an?\s+)?(?:'+NUMBER+AGE_UNIT+r'\s+)?(?:white\s+|black\s+)?(female|male)\b(?![-/]|\s+to\b)',re.I)
    else:
        pattern=re.compile(r'\bthe patient (?:identifies as|gender is)\s+(?:a\s+)?(woman|man|nonbinary|non-binary)\b(?![-/]|\s+to\b)',re.I)
    return [_fact(text,normalized_identity(m.group(1)),m.start(1),m.end(1),'summary_patient_'+dimension) for m in pattern.finditer(text)]


def address_evidence(text):
    result=[]
    for turn in turns(text):
        if turn['role']!='doctor':continue
        # Address to patient only: exclude a "sir" spoken by the patient to doctor.
        if OTHER_PERSON.search(turn['body']):continue
        for m in HONORIFIC.finditer(turn['body']):
            word=m.group(1).lower().rstrip('.')
            value='masculine' if word in {'sir','mister','mr'} else 'feminine'
            result.append(_fact(text,value,turn['start'],turn['body_start']+m.end(),'gendered_address_only'))
    return _unique(result)
