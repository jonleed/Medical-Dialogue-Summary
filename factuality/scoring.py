"""Validated annotation aggregation, separate by reviewer kind and evidence view."""
from .io import case_hash

KINDS = ('human', 'agent', 'model', 'synthetic')
TYPES = {'age','sex','gender','pronoun','race_ethnicity','medication','diagnosis','symptom','quantity','date','other'}
SUBJECTS = {'patient','relative','other'}
CLAIM_LABELS = {'supported','contradicted','unsupported','unclear','pending'}
FACT_LABELS = {'present','omitted','unclear','pending','excluded'}
ASSESSABLE = {'supported','contradicted','unsupported','present','omitted'}


def _require(condition, message):
    if not condition: raise ValueError(message)


def validate_span(text, span):
    _require(isinstance(span, dict), 'span must be an object')
    start, end = span.get('start'), span.get('end')
    _require(type(start) is int and type(end) is int and 0 <= start < end <= len(text), 'span offsets out of bounds')
    _require(text[start:end] == span.get('text'), 'span text does not match evidence')


def validate_annotation(case, annotation):
    a = annotation
    _require(a.get('case_id') == case['case_id'], 'case ID mismatch')
    _require(a.get('case_hash') == case_hash(case), 'stale case/prediction/evidence hash')
    _require(a.get('reviewer_kind') in KINDS, 'invalid reviewer kind')
    _require(a.get('status') == 'pending' or bool(a.get('reviewer', '').strip()), 'reviewer required')
    _require(a.get('status') in {'pending','partial','complete'}, 'invalid annotation status')
    _require(isinstance(a.get('unresolved'), list), 'unresolved must be a list')
    for field in ['claims_complete','expected_facts_complete','expected_facts_frozen','expected_facts_blind']:
        _require(type(a.get(field)) is bool, f'{field} must be boolean')
    for field in ['claims','expected_facts']:
        _require(isinstance(a.get(field), list), f'{field} must be a list')
    if a['status'] == 'pending':
        _require(not a['claims'] and not a['expected_facts'], 'pending annotations must have no final items; use partial')
    if a['expected_facts_complete'] or a['expected_facts']:
        _require(a.get('expected_facts_reviewer_kind') == a['reviewer_kind'], 'expected-fact provenance must match scoring reviewer kind')
        _require(a['expected_facts_frozen'] and a['expected_facts_blind'], 'expected-fact plan must be frozen and blind')
        _require(bool(a.get('expected_facts_reviewer')) and bool(a.get('expected_facts_frozen_at')), 'expected-fact reviewer/time required')
    if a['status'] != 'pending':
        _require(case.get('prediction') is not None, 'cannot review claims without generated prediction')
    if a['claims_complete'] and (case.get('prediction') or '').strip():
        _require(bool(a['claims']), 'nonempty prediction requires atomic claims')
    if a['status'] == 'complete':
        _require(a['claims_complete'] and a['expected_facts_complete'], 'complete status requires both review passes')
    for field, labels in [('claims', CLAIM_LABELS), ('expected_facts', FACT_LABELS)]:
        seen = set()
        propositions = set()
        for item in a[field]:
            _require(isinstance(item.get('id'), str) and item['id'] not in seen, 'item IDs must be unique strings')
            seen.add(item['id'])
            _require(item.get('label') in labels, 'invalid label')
            _require(item.get('fact_type') in TYPES, 'invalid fact type')
            _require('subject' not in item or item['subject'] in SUBJECTS, 'invalid subject')
            _require(bool(item.get('text', '').strip()), 'atomic text required')
            _require(item.get('visibility') in {'visible','truncated','unclear'}, 'invalid visibility')
            _require(isinstance(item.get('source_spans'), list) and isinstance(item.get('visible_spans'), list), 'evidence span lists required')
            for evidence in item['source_spans']: validate_span(case['dialogue'], evidence)
            for evidence in item['visible_spans']: validate_span((case.get('model_input') or {}).get('text', ''), evidence)
            if field == 'claims':
                validate_span(case.get('prediction') or '', item.get('generated_span'))
                key = (item['generated_span']['start'],item['generated_span']['end'],item['fact_type'],item['text'])
                _require(key not in propositions, 'duplicate atomic claim')
                propositions.add(key)
            elif item['label'] == 'present': validate_span(case.get('prediction') or '', item.get('generated_span'))
            if item['label'] not in {'pending','unclear'}:
                _require(bool(item.get('rationale','').strip()), 'final label requires rationale')
            if item['label'] in {'supported','contradicted','present','omitted'}:
                _require(bool(item['source_spans']), 'positive original source evidence required')
                if item['visibility'] == 'visible': _require(bool(item['visible_spans']), 'model-visible evidence required')
    return a


def _metric(numerator, denominator):
    return {'numerator':numerator, 'denominator':denominator,
            'rate':numerator / denominator if denominator else None}


def _view(annotations, visible):
    claims, facts, eligible_cases, error_cases = [], [], set(), set()
    for a in annotations:
        for field in ['claims', 'expected_facts']:
            for item in a[field]:
                if item['label'] not in ASSESSABLE: continue
                if visible and item['visibility'] != 'visible': continue
                (claims if field == 'claims' else facts).append(item)
                eligible_cases.add(a['case_id'])
                if item['label'] in {'contradicted','unsupported','omitted'}: error_cases.add(a['case_id'])
    def contradictions(types):
        subset = [i for i in claims if i['fact_type'] in types]
        return _metric(sum(i['label']=='contradicted' for i in subset),len(subset))
    return {'assessable_claims':len(claims), 'age_contradictions':contradictions({'age'}),
        'gender_reference_contradictions':contradictions({'sex','gender','pronoun'}),
        'sex_contradictions':contradictions({'sex'}), 'gender_identity_contradictions':contradictions({'gender'}),
        'pronoun_contradictions':contradictions({'pronoun'}),
        'contradicted_claims':_metric(sum(i['label']=='contradicted' for i in claims),len(claims)),
        'unsupported_claims':_metric(sum(i['label']=='unsupported' for i in claims),len(claims)),
        'expected_fact_omissions':_metric(sum(i['label']=='omitted' for i in facts),len(facts)),
        'expected_fact_omissions_by_type':{typ:_metric(sum(i['label']=='omitted' and i['fact_type']==typ for i in facts),sum(i['fact_type']==typ for i in facts)) for typ in sorted(TYPES)},
        'summaries_with_error':_metric(len(error_cases),len(eligible_cases))}


def score(cases, annotations):
    _require(len({c['case_id'] for c in cases}) == len(cases), 'duplicate case IDs')
    _require(len({c.get('dataset_kind') for c in cases}) <= 1, 'cannot mix synthetic and empirical cases')
    _require(all(c.get('dataset_kind') in {'real','synthetic'} for c in cases), 'invalid dataset kind')
    _require(len({c.get('protocol') for c in cases}) <= 1, 'cannot mix generation protocols')
    by_id = {c['case_id']:c for c in cases}
    seen = set()
    for a in annotations:
        _require(a.get('case_id') in by_id, 'unknown annotation case')
        validate_annotation(by_id[a['case_id']], a)
        key = (a['case_id'],a['reviewer_kind'])
        _require(key not in seen, 'duplicate annotations; adjudicate independently before scoring')
        seen.add(key)
        _require((by_id[a['case_id']].get('dataset_kind') == 'synthetic') == (a['reviewer_kind'] == 'synthetic'), 'synthetic labels require synthetic cases')
    result = {'schema_version':1, 'dataset_kind':cases[0].get('dataset_kind') if cases else None,
        'cases':len(cases), 'generated_cases':sum(c.get('prediction') is not None for c in cases),
        'truncated_cases':sum(bool((c.get('model_input') or {}).get('truncated')) for c in cases)}
    full_tokens=sum((c.get('model_input') or {}).get('full_token_count',0) for c in cases)
    visible_tokens=sum(sum((c.get('model_input') or {}).get('attention_mask',[])) for c in cases)
    result['input_token_coverage']={'full_tokens':full_tokens,'visible_tokens':visible_tokens,
        'discarded_tokens':full_tokens-visible_tokens if full_tokens else None,
        'visible_fraction':visible_tokens/full_tokens if full_tokens else None,
        'note':'Input token coverage, not clinically relevant fact coverage.'}
    from .checks import demographic_suggestions, automatic_summary
    suggestions = [s for c in cases for s in demographic_suggestions(c)]
    result['automatic'] = automatic_summary(suggestions)
    for kind in KINDS:
        group = [a for a in annotations if a['reviewer_kind']==kind and a['status']!='pending']
        def fully_resolved(a):
            return a['status']=='complete' and not a['unresolved'] and all(i['label'] not in {'pending','unclear'} for f in ['claims','expected_facts'] for i in a[f])
        complete = [a for a in group if fully_resolved(a)]
        result[kind] = {'cases_with_annotations':len(group), 'fully_reviewed_cases':len(complete),
          'partial_or_unresolved_cases':len(group)-len(complete), 'pending_cases':len(cases)-len(group),
          'review_coverage':_metric(len(group),len(cases)),
          'passing_complete_cases':sum(not any(i['label'] in {'contradicted','unsupported','omitted'} for f in ['claims','expected_facts'] for i in a[f]) for a in complete),
          'unclear_claims':sum(i['label']=='unclear' for a in group for i in a['claims']),
          'pending_claims':sum(i['label']=='pending' for a in group for i in a['claims']),
          'unclear_expected_facts':sum(i['label']=='unclear' for a in group for i in a['expected_facts']),
          'pending_expected_facts':sum(i['label']=='pending' for a in group for i in a['expected_facts']),
          'excluded_expected_facts':sum(i['label']=='excluded' for a in group for i in a['expected_facts']),
          'visibility_unclear_items':sum(i['visibility']=='unclear' for a in group for f in ['claims','expected_facts'] for i in a[f]),
          'truncated_expected_facts':sum(i['visibility']=='truncated' and i['label'] in {'present','omitted'} for a in group for i in a['expected_facts']),
          'truncated_omitted_facts':sum(i['visibility']=='truncated' and i['label']=='omitted' for a in group for i in a['expected_facts']),
          'full_source':_view(group,False), 'model_visible':_view(group,True)}
        result[kind]['expected_fact_visibility_coverage'] = _metric(result[kind]['model_visible']['expected_fact_omissions']['denominator'],result[kind]['full_source']['expected_fact_omissions']['denominator'])
        result[kind]['claim_visibility_coverage'] = _metric(result[kind]['model_visible']['assessable_claims'],result[kind]['full_source']['assessable_claims'])
    return result
