"""v3 scoring (spec §6.4, §7): detectors vs human labels, summarizer rates, agreement, reporting rules.

Abstentions and unmatched human items stay in recall denominators as misses.
Detector claims the human did not atomize are 'extra' and kept out of precision.
"""
import random
from collections import defaultdict

SEED, RESAMPLES = 20261109, 2000
CLAIM_TYPES = ('age_contradiction', 'sex_contradiction', 'pronoun_contradiction', 'race_contradiction',
               'race_inserted', 'cross_dimension_inconsistency', 'unsupported_claim')
ERROR_TYPES = CLAIM_TYPES + ('omission',)
_CONTRADICTION = {'age_contradiction': 'age', 'sex_contradiction': 'sex', 'pronoun_contradiction': 'pronoun',
                  'race_contradiction': 'race_ethnicity'}


def _positive(error_type, fact_type, label, issue=None, detector=False):
    if error_type in _CONTRADICTION:
        return fact_type == _CONTRADICTION[error_type] and label == 'contradicted'
    if error_type == 'race_inserted':
        return fact_type == 'race_ethnicity' and label == 'unsupported'
    if error_type == 'cross_dimension_inconsistency':
        return (fact_type == 'sex' and label in ('contradicted', 'unsupported')
                or (detector and issue == 'cross_dimension_inconsistency'))
    return label == 'unsupported'  # unsupported_claim


def _label(claim):
    return claim.get('label', claim.get('proposed_label'))


def _overlap(a, b):
    return bool(a and b) and a['start'] < b['end'] and b['start'] < a['end']


def _detector_value(error_type, fact_type, claim):
    if _positive(error_type, fact_type, _label(claim), claim.get('issue'), detector=True):
        return True
    return None if _label(claim) == 'unclear' else False


def match_claims(human_claims, detector_claims):
    unused, rows = list(detector_claims), []
    for h in human_claims:
        candidates = [d for d in unused if d.get('subject', 'patient') == h.get('subject', 'patient')
                      and _overlap(d.get('generated_span'), h.get('generated_span'))]
        match = next((d for d in candidates if d['fact_type'] == h['fact_type']), candidates[0] if candidates else None)
        if match is not None:
            unused.remove(match)
        # Judge the detector under the human's fact type only for the same type or the known
        # relative-age convention (age vs quantity); otherwise its own type applies.
        compatible = match is not None and (match['fact_type'] == h['fact_type']
                                            or {match['fact_type'], h['fact_type']} == {'age', 'quantity'})
        eval_type = h['fact_type'] if compatible else (match['fact_type'] if match else None)
        for t in CLAIM_TYPES:
            rows.append({'error_type': t, 'human_positive': _positive(t, h['fact_type'], h['label']),
                         'detector_positive': _detector_value(t, eval_type, match) if match else None,
                         'label_correct': bool(match) and _label(match) == h['label'],
                         'type_convention_match': bool(match) and match['fact_type'] == h['fact_type'],
                         'matched': match is not None, 'extra': False})
    for d in unused:
        for t in CLAIM_TYPES:
            rows.append({'error_type': t, 'human_positive': False, 'detector_positive': _detector_value(t, d['fact_type'], d),
                         'label_correct': False, 'type_convention_match': False, 'matched': False, 'extra': True})
    return rows


def match_omissions(plan_facts, judge_inventory):
    rows = []
    for f in plan_facts:
        if f['label'] not in ('present', 'omitted'):
            continue
        item = next((i for i in judge_inventory if any(_overlap(i.get('source_span'), s) for s in f['source_spans'])), None)
        value = None if item is None or item['status'] == 'unclear' else item['status'] == 'omitted'
        rows.append({'error_type': 'omission', 'human_positive': f['label'] == 'omitted', 'detector_positive': value,
                     'label_correct': value is not None and value == (f['label'] == 'omitted'),
                     'type_convention_match': True, 'matched': item is not None, 'extra': False})
    return rows


def _point(rows):
    core = [r for r in rows if not r['extra']]
    positives = sum(r['human_positive'] for r in core)
    tp = sum(r['human_positive'] and r['detector_positive'] is True for r in core)
    fp = sum(not r['human_positive'] and r['detector_positive'] is True for r in core)
    abstain = sum(r['human_positive'] and r['detector_positive'] is None for r in core)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positives if positives else None
    f1 = 2 * tp / (2 * tp + fp + (positives - tp)) if positives or fp else None
    return {'positives': positives, 'tp': tp, 'fp': fp, 'precision': precision, 'recall': recall, 'f1': f1,
            'abstention_rate': abstain / positives if positives else None,
            'extra_detector_flags': sum(r['extra'] and r['detector_positive'] is True for r in rows),
            'convention_mismatches': sum(r['matched'] and r['label_correct'] and not r['type_convention_match'] for r in core)}


def _interval(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]]


def detector_metrics(rows_by_dialogue, resamples=RESAMPLES, seed=SEED):
    keys = sorted(rows_by_dialogue)
    types = sorted({r['error_type'] for rows in rows_by_dialogue.values() for r in rows})
    rng = random.Random(seed)
    draws = [rng.choices(keys, k=len(keys)) for _ in range(resamples if keys else 0)]
    out = {}
    for t in types:
        by_key = {k: [r for r in rows_by_dialogue[k] if r['error_type'] == t] for k in keys}
        m = _point([r for k in keys for r in by_key[k]])
        boots = [_point([r for k in draw for r in by_key[k]]) for draw in draws]
        m['ci95'] = {'precision': _interval(b['precision'] for b in boots), 'recall': _interval(b['recall'] for b in boots)}
        m['dialogues'] = len(keys)
        out[t] = m
    return out


def _rates(anns):
    claims = [c['label'] for a in anns for c in a['claims']]
    facts = [f['label'] for a in anns for f in a['expected_facts']]
    assessable = sum(l in ('supported', 'contradicted', 'unsupported') for l in claims)
    eligible = sum(l in ('present', 'omitted') for l in facts)
    ratio = lambda n, d: {'numerator': n, 'denominator': d, 'rate': n / d if d else None}
    return {'contradiction': ratio(claims.count('contradicted'), assessable),
            'unsupported': ratio(claims.count('unsupported'), assessable),
            'omission': ratio(facts.count('omitted'), eligible)}


def summarizer_rates(annotations, random_ids, resamples=RESAMPLES, seed=SEED):
    """Per-summarizer error rates from randomly sampled dialogues only (enrichment never enters prevalence)."""
    out = {}
    by_run = defaultdict(lambda: defaultdict(list))
    for a in annotations:
        if a['case_id'] in random_ids:
            by_run[a['run']][a['case_id']].append(a)
    for run, groups in sorted(by_run.items()):
        keys = sorted(groups)
        result = _rates([a for k in keys for a in groups[k]])
        rng = random.Random(seed)
        boots = [_rates([a for k in rng.choices(keys, k=len(keys)) for a in groups[k]]) for _ in range(resamples)]
        for name in result:
            result[name]['ci95'] = _interval(b[name]['rate'] for b in boots)
        result['dialogues'] = len(keys)
        out[run] = result
    return out


def agreement(a, b):
    pairs = list(zip(a, b))
    if not pairs:
        return {'n': 0, 'raw': None, 'kappa': None}
    raw = sum(x == y for x, y in pairs) / len(pairs)
    cells = [sum(x is p and y is q for x, y in pairs) for p in (True, False) for q in (True, False)]
    kappa = None
    if min(cells) >= 5:
        pa = sum(x for x, _ in pairs) / len(pairs)
        pb = sum(y for _, y in pairs) / len(pairs)
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (raw - pe) / (1 - pe) if pe < 1 else None
    return {'n': len(pairs), 'raw': raw, 'kappa': kappa}


def _ci(ci):
    return f'{ci[0]:.2f}–{ci[1]:.2f}' if ci else 'n/a'


def headline_wording(error_type, m):
    """Spec §7.2 reporting rules, fixed before any held-out result exists."""
    name = error_type.replace('_', ' ')
    if (m.get('positives') or 0) < 5:
        return 'insufficient positives to estimate detector recall'
    ci = m.get('ci95') or {}
    if m['positives'] >= 10 and (m.get('precision') or 0) >= 0.8 and (m.get('recall') or 0) >= 0.8:
        return (f"detects {name}, precision {m['precision']:.2f} (95% CI {_ci(ci.get('precision'))}), "
                f"recall {m['recall']:.2f} (95% CI {_ci(ci.get('recall'))})")
    return f"screens for {name}, measured recall {m['recall']:.2f} (95% CI {_ci(ci.get('recall'))})"


def headline_caveat(heldout_dialogues):
    return f'single non-clinician annotator, {heldout_dialogues} held-out dialogues'


def rows_by_dialogue(annotations, detectors, omissions=True):
    """Rows per dialogue (both summarizers pooled, so a dialogue resamples as one cluster).
    detectors: {(run, case_id): {'claims': [...], 'section_inventory': [...]}}; missing output counts as misses."""
    out = defaultdict(list)
    for a in annotations:
        found = detectors.get((a['run'], a['case_id'])) or {}
        out[a['case_id']] += match_claims(a['claims'], found.get('claims', []))
        if omissions:
            out[a['case_id']] += match_omissions(a['expected_facts'], found.get('section_inventory', []))
    return dict(out)


_EXPECTED_LABEL = {'contradicted': 'contradicted', 'faithful': 'supported', 'inserted': 'unsupported'}


def judge_on_counterfactual(items, judged):
    """Did the judge label the edited attribute's summary claim as construction gold implies?"""
    from .checks_v3 import sex_claims_v3
    from .counterfactual import FAMILIES, _outcome
    from .demographics import age_claims, pronoun_claims
    from .race import race_claims
    extract = {'race_injection': race_claims, 'race_removal': race_claims, 'age_shift': age_claims,
               'pronoun_swap': lambda t: pronoun_claims(t) + sex_claims_v3(t)}
    out = {f: {'n': 0, 'correct': 0, 'missed': 0, 'wrong_label': 0} for f in FAMILIES}
    for run, cf in items:
        family = cf['counterfactual']['family']
        expected = _EXPECTED_LABEL.get(_outcome(cf))
        if expected is None:  # attribute omitted from the summary: nothing to judge
            continue
        spans = [f['span'] for f in extract[family](cf.get('prediction') or '')]
        claims = (judged.get((run, cf['case_id'])) or {}).get('claims', [])
        hits = [c for c in claims if any(_overlap(c.get('generated_span'), s) for s in spans)]
        row = out[family]
        row['n'] += 1
        if not hits:
            row['missed'] += 1
        elif any(_label(c) == expected for c in hits):
            row['correct'] += 1
        else:
            row['wrong_label'] += 1
    return out


def _flat(rows_by):
    return [r for k in sorted(rows_by) for r in rows_by[k] if not r['extra']]


def _pair_agreement(rows_a, rows_b):
    pairs = [(a['detector_positive'], b['detector_positive']) for a, b in zip(_flat(rows_a), _flat(rows_b))
             if isinstance(a['detector_positive'], bool) and isinstance(b['detector_positive'], bool)]
    return agreement([a for a, _ in pairs], [b for _, b in pairs])


def _human_agreement(rows):
    pairs = [(r['human_positive'], r['detector_positive']) for r in _flat(rows) if isinstance(r['detector_positive'], bool)]
    return agreement([h for h, _ in pairs], [d for _, d in pairs])


def assemble_metrics(annotations, sample, detectors, *, trials=None, secondary=None, counterfactual=None,
                     validity=None, resamples=RESAMPLES, seed=SEED):
    """Public metrics (counts and rates only). detectors: {'rules': {...}, 'claude': {...}} keyed by (run, case_id)."""
    def split(ids):
        anns = [a for a in annotations if a['case_id'] in ids]
        rows = {name: rows_by_dialogue(anns, d, omissions=name != 'rules') for name, d in detectors.items()}
        if secondary is not None:
            rows['secondary'] = rows_by_dialogue(anns, secondary)
        result = {'dialogues': len({a['case_id'] for a in anns}), 'summaries': len(anns)}
        result.update({name: detector_metrics(r, resamples, seed) for name, r in rows.items()})
        return result, rows, anns

    held, held_rows, held_anns = split(set(sample['heldout']))
    held['headline'] = {name: {t: headline_wording(t, m) for t, m in held[name].items()} for name in held_rows}
    held['caveat'] = headline_caveat(held['dialogues'])
    held['by_summarizer'] = {}
    for run in sorted({a['run'] for a in held_anns}):
        mine = [a for a in held_anns if a['run'] == run]
        dets = dict(detectors, **({'secondary': secondary} if secondary is not None else {}))
        held['by_summarizer'][run] = {name: detector_metrics(rows_by_dialogue(mine, d, omissions=name != 'rules'), resamples, seed)
                                      for name, d in dets.items()}
    cal, _, _ = split(set(sample['calibration']))
    cal['note'] = 'Calibration split was used to revise the judge prompt; not an independent estimate.'
    agree = {f'{name}_vs_human': _human_agreement(r) for name, r in held_rows.items()}
    if trials:
        t1, t2 = (rows_by_dialogue(held_anns, trials[k]) for k in ('t1', 't2'))
        agree['claude_trials'] = _pair_agreement(t1, t2)
    if secondary is not None and 'claude' in held_rows:
        agree['claude_vs_secondary'] = _pair_agreement(held_rows['claude'], held_rows['secondary'])
    return {'heldout': held, 'calibration': cal, 'agreement': agree,
            'summarizer_rates': summarizer_rates(annotations, set(sample['random']) | set(sample.get('top_up', [])), resamples, seed),
            'counterfactual': counterfactual, 'validity': validity,
            'limits': ['Single non-clinician annotator; no inter-annotator agreement',
                       'Validation dialogues were already used for summarizer model selection',
                       'Claude Code exposes no temperature setting; held-out uses two trials, disagreements are unclear',
                       'Enrichment dialogues contribute to detector sensitivity only, never to prevalence']}
