"""One-time migration of the reviewer labels to the merged gender-reference rule.

Sex terms and pronouns are one dimension (the patient's gender reference). Evidence is an explicit
statement, a third-person pronoun that refers to the patient, or a gendered form of address directed at
the patient; claims that match it are supported, claims that conflict with it are contradicted, and
claims with no evidence are unsupported. The original labels already followed that rule almost everywhere,
so this lists only the exceptions, found by comparing every gender-reference label with the rule checker.

Run it once, against the original bundle, to produce the revised one:

    git archive a078f7c data/factuality_benchmark | tar -x -C /tmp/original
    python scripts/relabel_gender_reference.py --src /tmp/original/data/factuality_benchmark --out /tmp/revised

Each edit asserts the state it expects, so running it on any other bundle fails instead of guessing.
The reviewer of record is unchanged: every edit below was checked by the reviewer.
"""
import argparse
import json
import shutil
from pathlib import Path

COMPACT = {'plans.jsonl': (',', ':')}  # plans.jsonl is written without spaces; keep each file's own style


def dump(row, name=''):
    return json.dumps(row, ensure_ascii=False, separators=COMPACT.get(name))


def load(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            assert dump(row, Path(path).name) == line, f'{path}: a line does not round-trip, untouched rows would change'
            rows.append(row)
    return rows


def spans_in(dialogue, visible, needle):
    """The needle as a source span and as the matching span in the text the model saw."""
    assert dialogue.count(needle) == 1, f'ambiguous or missing needle {needle!r}'
    start = dialogue.index(needle)
    offset = visible.find(dialogue[:60])
    assert offset >= 0 and visible[offset + start:offset + start + len(needle)] == needle, 'span not model-visible'
    return ({'start': start, 'end': start + len(needle), 'text': needle},
            {'start': offset + start, 'end': offset + start + len(needle), 'text': needle})


def main(src_dir, out_dir):
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    assert not out_dir.exists(), f'{out_dir} already exists'
    shutil.copytree(src_dir, out_dir)
    cases = {c['case_id']: c for c in load(src_dir / 'cases.jsonl')}
    summaries = {(s['model'], s['case_id']): s for s in load(src_dir / 'summaries.jsonl')}
    labels = load(src_dir / 'labels.jsonl')
    plans = load(src_dir / 'plans.jsonl')
    by_key = {(l['model'], l['case_id']): l for l in labels}
    log = []

    def claim(model, case, cid):
        return next(c for c in by_key[(model, case)]['claims'] if c['id'] == cid)

    def fact(model, case, fid):
        return next(f for f in by_key[(model, case)]['expected_facts'] if f['id'] == fid)

    def visible(model, case):
        return summaries[(model, case)]['model_input']['text']

    def respan(item, model, case, needle):
        item['source_spans'], item['visible_spans'] = [[s] for s in spans_in(cases[case]['dialogue'], visible(model, case), needle)]

    # 1. Evidence that does not establish the claim.
    c = claim('biobart-large-lora', 'val:32', 'c1')
    assert c['label'] == 'supported' and c['source_spans'][0]['text'] == 'Thank you, he was only fifty two.'
    respan(c, 'biobart-large-lora', 'val:32', "I'm so sorry to hear that, sir.")
    c['rationale'] = "The clinician addresses the patient as 'sir'. The V1 evidence ('he was only fifty two') is about the patient's father."
    log.append(('val:32', 'biobart-large-lora', 'c1 Patient is male', 'supported (span about the father)', 'supported (span: "sir")'))

    c = claim('bart-base-lora', 'val:94', 'c1')
    assert c['label'] == 'supported' and c['source_spans'][0]['text'] == 'Mister Jay'
    respan(c, 'bart-base-lora', 'val:94', 'I reviewed his past medical history')
    c['rationale'] = "The clinician refers to the patient as 'his' (third person)."
    log.append(('val:94', 'bart-base-lora', 'c1 Patient is male (his)', 'supported (span: "Mister Jay")', 'supported (span: "I reviewed his past medical history")'))

    c = claim('bart-base-lora', 'val:36', 'c2')
    assert c['label'] == 'supported' and c['source_spans'][0]['text'] == 'I had a negative pregnancy test.'
    c['label'] = 'unclear'
    c['rationale'] = ("A pregnancy test hints at the patient's sex but is not an explicit statement, a patient pronoun or "
                      "gendered address, so the claim has no firm evidence either way.")
    log.append(('val:36', 'bart-base-lora', 'c2 Patient is female (she/her)', 'supported', 'unclear'))

    # 2. Patient pronouns the review never recorded.
    def add_pronoun_claim(model, case, text, needle, gen_needle, gen_from, why):
        l = by_key[(model, case)]
        assert not any(c['fact_type'] in ('sex', 'pronoun') for c in l['claims']), 'gender-reference claim already present'
        src, vis = spans_in(cases[case]['dialogue'], visible(model, case), needle)
        pred = summaries[(model, case)]['prediction']
        start = pred.index(gen_needle, pred.index(gen_from))
        new = {'id': f"c{max(int(c['id'][1:]) for c in l['claims']) + 1}", 'label': 'supported', 'fact_type': 'pronoun',
               'subject': 'patient', 'visibility': 'visible', 'text': text, 'rationale': why,
               'source_spans': [src], 'visible_spans': [vis],
               'generated_span': {'start': start, 'end': start + len(gen_needle), 'text': gen_needle}}
        l['claims'].append(new)
        log.append((case, model, f"{new['id']} {text}", '(missing)', 'supported'))

    add_pronoun_claim('biobart-large-lora', 'val:83', 'Patient is male (he)', 'please give him Norflex sixty M G',
                      'He has been given', 'He has been given', "The clinician says 'give him' about the patient.")
    ref = claim('bart-base-lora', 'val:62', 'c2')
    assert ref['source_spans'][0]['text'].startswith('This is Miss A B C')
    add_pronoun_claim('biobart-large-lora', 'val:62', 'Patient is female (she/her)', ref['source_spans'][0]['text'],
                      'her', 'of her ', "The clinician refers to the patient as 'she'.")

    # 3. val:51: the plan span "you need it." does not establish "substance abuse".
    needle = 'try to stop your substance abuse'
    for m in ('bart-base-lora', 'biobart-large-lora'):
        f = fact(m, 'val:51', 'f1')
        assert f['source_spans'][0]['text'] == 'you need it.'
        respan(f, m, 'val:51', needle)
    pf = next(f for p in plans if p['case_id'] == 'val:51' for f in p['expected_facts'] if f['id'] == 'f1')
    assert pf['source_spans'][0]['text'] == 'you need it.'
    src, _ = spans_in(cases['val:51']['dialogue'], cases['val:51']['dialogue'], needle)
    pf['source_spans'], pf['visible_spans'] = [src], [dict(src)]
    fact('bart-base-lora', 'val:51', 'f1')['rationale'] = 'The summary names substance abuse, which the clinician asks the patient to stop.'
    log.append(('val:51', 'bart-base-lora', 'f1 Substance abuse', 'present (span: "you need it.")', 'present (span: "try to stop your substance abuse")'))
    f = fact('biobart-large-lora', 'val:51', 'f1')
    assert f['label'] == 'present'
    f['label'] = 'unclear'
    f['rationale'] = ("The summary says 'history of' substance abuse, but the abuse is ongoing; claim c1 is unclear for the "
                      "same reason, so the fact is unclear too.")
    log.append(('val:51', 'biobart-large-lora', 'f1 Substance abuse', 'present (span: "you need it.")', 'unclear (span: "try to stop your substance abuse")'))

    # 4. val:53: "Denies cough" is borderline against "cold like symptoms".
    f = fact('biobart-large-lora', 'val:53', 'f9')
    assert f['label'] == 'omitted'
    f['label'] = 'unclear'
    f['rationale'] = ("The patient answered 'No' to a list that includes cough; the summary says 'denied any fever, sore "
                      "throat, or any cold like symptoms' without naming cough. Borderline, so unclear.")
    log.append(('val:53', 'biobart-large-lora', 'f9 Denies cough', 'omitted', 'unclear'))

    (out_dir / 'labels.jsonl').write_text(''.join(dump(r) + '\n' for r in labels), encoding='utf-8')
    (out_dir / 'plans.jsonl').write_text(''.join(dump(r, 'plans.jsonl') + '\n' for r in plans), encoding='utf-8')
    held = set(json.loads((src_dir / 'splits.json').read_text())['heldout'])
    lines = ['# Label revision', '',
             'Generated by `scripts/relabel_gender_reference.py`. Summaries, judge outputs, cases and counterfactuals are '
             'unchanged. The reviewer of record is unchanged; each edit was checked by the reviewer.', '',
             '| Split | Case | Model | Item | Before | After |', '|---|---|---|---|---|---|']
    for case, model, what, before, after in sorted(log, key=lambda r: (r[0] not in held, r[0], r[1], r[2])):
        lines.append(f"| {'held-out' if case in held else 'calibration'} | {case} | {model} | {what} | {before} | {after} |")
    (out_dir / 'CHANGES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {out_dir}: {len(log)} changes')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    main(a.src, a.out)
