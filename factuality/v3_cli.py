"""v3 factuality pipeline stages: python -m factuality.v3_cli <stage> --help."""
import argparse
import json
from pathlib import Path

from .io import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'results' / 'factuality' / 'local'
V3 = LOCAL / 'v3'
RUNS = [
    ('biobart-flask-2026-10-17', 'BioBART-large with LoRA'),
    ('bart-lora-flask-2026-10-17', 'BART-base with LoRA'),
    ('bart-base-flask-2026-10-17', 'BART-base, no adapter'),
]
FINE_TUNED = RUNS[:2]
STAGES = {}


def stage(name, help_text):
    def register(fn):
        STAGES[name] = (fn, help_text)
        return fn
    return register


def load_run(run_dir):
    return read_jsonl(Path(run_dir) / 'cases.jsonl')


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


@stage('validity', 'Classify saved outputs as summary/prompt_echo/empty/degenerate')
def cmd_validity(args):
    from .validity import validity_summary
    report = {}
    for run in args.runs:
        s = validity_summary(load_run(run))
        report[Path(run).name] = s
        print(Path(run).name, s['counts'])
    dump(args.out, report)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='stage', required=True)
    v = subs.add_parser('validity', help=STAGES['validity'][1])
    v.add_argument('--runs', nargs='+', default=[str(LOCAL / r) for r, _ in RUNS])
    v.add_argument('--out', default=str(V3 / 'validity.json'))
    for extend in EXTENSIONS:
        extend(subs)
    return p


EXTENSIONS = []  # later stages append parser-builders here


def _screen_parser(subs):
    sp = subs.add_parser('screen', help=STAGES['screen'][1])
    sp.add_argument('--out', required=True, help='public aggregate path, e.g. results/factuality/v3-<date>/screening.json')
    sp.add_argument('--local-out', default=str(V3 / 'screen_per_case.jsonl'))


@stage('screen', 'Validity-gated v3 rule screening of the saved runs')
def cmd_screen(args):
    from .io import write_jsonl
    from .v3_screen import per_case_local, screen_runs
    runs = [(r, label, load_run(LOCAL / r)) for r, label in RUNS]
    report = screen_runs(runs)
    dump(args.out, report)
    local = Path(args.local_out)
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        local.unlink()
    write_jsonl(local, per_case_local(runs))
    for m in report['models']:
        a = m['automatic']
        print(f"{m['model']:<26} valid {m['valid_cases']:>3}/{m['cases']}  suggestions {a['suggestions']:>3}  "
              f"contradicted {a['contradiction_suggestions']}  unsupported {a['unsupported_suggestions']}  "
              f"unclear {a['unclear_suggestions']}  flagged cases {a['flagged_cases']}")


EXTENSIONS.append(_screen_parser)


TOKENIZERS = ('GanjinZero/biobart-large', 'facebook/bart-base')  # app.py loads the tokenizer from the base model


def _cf_parsers(subs):
    b = subs.add_parser('cf-build', help=STAGES['cf-build'][1])
    b.add_argument('--cases', default=str(LOCAL / RUNS[0][0] / 'cases.jsonl'))
    b.add_argument('--out', default=str(V3 / 'cf_cases.jsonl'))
    b.add_argument('--cap', type=int, default=250)
    sc = subs.add_parser('cf-score', help=STAGES['cf-score'][1])
    sc.add_argument('--runs', nargs='+', required=True, help='generation output directories (cases.jsonl inside)')
    sc.add_argument('--out', required=True)


@stage('cf-build', 'Build the counterfactual packet from validation cases (no predictions printed)')
def cmd_cf_build(args):
    from collections import Counter
    from transformers import AutoTokenizer
    from .counterfactual import build_packet
    from .io import write_jsonl
    toks = [AutoTokenizer.from_pretrained(t, local_files_only=True) for t in TOKENIZERS]
    count = lambda s: max(len(t(s)['input_ids']) for t in toks)
    packet = build_packet(read_jsonl(args.cases), count, cap=args.cap)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f'{out} exists; use a new path (packets are frozen once built)')
    write_jsonl(out, packet)
    print(dict(Counter(c['counterfactual']['family'] for c in packet)), 'total', len(packet))
    for c in packet:
        if c['counterfactual']['family'] == 'race_removal':
            for e in c['counterfactual']['edits']:
                base = next(x for x in read_jsonl(args.cases) if x['case_id'] == c['counterfactual']['base_case_id'])
                print('REMOVAL', c['case_id'], 'before:', repr(base['dialogue'][max(0, e['start'] - 80):e['end'] + 40]))
                print('        after: ', repr(c['dialogue'][max(0, e['start'] - 80):e['start'] + 40]))


@stage('cf-score', 'Score counterfactual generations against construction gold (rules side)')
def cmd_cf_score(args):
    from .counterfactual import score_counterfactual
    report = {Path(r).name: score_counterfactual(in_packet(read_jsonl(Path(r) / 'cases.jsonl'))) for r in args.runs}
    dump(args.out, report)
    for name, r in report.items():
        print(name, {k: v for k, v in r['families'].items()})


EXTENSIONS.append(_cf_parsers)


def _sample_parser(subs):
    sp = subs.add_parser('sample', help=STAGES['sample'][1])
    sp.add_argument('--screen', default=str(V3 / 'screen_per_case.jsonl'))
    sp.add_argument('--out', default=str(V3 / 'sample.json'))


@stage('sample', 'Seeded annotation sample and splits (prints sizes only, never which issues qualified)')
def cmd_sample(args):
    from .sampling import enrichment_eligible, select_sample
    rows = read_jsonl(args.screen)
    ids = sorted({c['case_id'] for c in load_run(LOCAL / RUNS[0][0])})
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists; the sample is frozen once drawn')
    sample = select_sample(ids, enrichment_eligible(rows, {r for r, _ in FINE_TUNED}))
    dump(out, sample)
    print({k: len(sample[k]) for k in ['random', 'enrichment', 'calibration', 'heldout']},
          'enrichment pool', sample['enrichment_pool_size'])


EXTENSIONS.append(_sample_parser)


def _sampled_cases(sample_path, run_dir):
    sample = json.loads(Path(sample_path).read_text())
    ids = set(sample['calibration']) | set(sample['heldout'])
    return [c for c in load_run(run_dir) if c['case_id'] in ids], sample


def _fine_tuned_sampled(sample_path):
    from .validity import classify_output
    out = {}
    for run, _ in FINE_TUNED:
        cases, _ = _sampled_cases(sample_path, LOCAL / run)
        out[run] = [c for c in cases if classify_output(c.get('prediction'), c['dialogue']) == 'summary']
    return out


def _annotation_parsers(subs):
    a = subs.add_parser('plans', help=STAGES['plans'][1])
    a.add_argument('--sample', default=str(V3 / 'sample.json'))
    a.add_argument('--out', default=str(V3 / 'plans'))
    f = subs.add_parser('freeze-plans', help=STAGES['freeze-plans'][1])
    f.add_argument('--plans', default=str(V3 / 'plans' / 'plans.jsonl'))
    lp = subs.add_parser('label-packet', help=STAGES['label-packet'][1])
    lp.add_argument('--sample', default=str(V3 / 'sample.json'))
    lp.add_argument('--plans', default=str(V3 / 'plans' / 'plans.jsonl'))
    lp.add_argument('--out', default=str(V3 / 'label'))
    il = subs.add_parser('import-labels', help=STAGES['import-labels'][1])
    il.add_argument('--labels', required=True, help='labels.jsonl exported from label.html')
    il.add_argument('--sample', default=str(V3 / 'sample.json'))
    il.add_argument('--key', default=str(V3 / 'label' / 'key.json'))
    il.add_argument('--out', default=str(V3 / 'annotations.jsonl'))


@stage('plans', 'Write the blind source-only plan packet (no predictions) for the sampled dialogues')
def cmd_plans(args):
    from .annotate import write_plan_packet
    cases, _ = _sampled_cases(args.sample, LOCAL / RUNS[0][0])
    out = write_plan_packet(cases, args.out)
    print(f'{len(cases)} plans -> {out / "plans.html"} (export plans.jsonl from the page into {out})')


@stage('freeze-plans', 'Validate and hash-freeze the completed plans')
def cmd_freeze_plans(args):
    from .annotate import freeze_plans
    print(freeze_plans(args.plans))


@stage('label-packet', 'Write the model-blinded labelling page for both fine-tuned runs')
def cmd_label_packet(args):
    from .annotate import write_label_packet
    out = write_label_packet(_fine_tuned_sampled(args.sample), args.plans, Path(args.plans).parent / 'plans_freeze.json', args.out)
    print(f'labelling page -> {out / "label.html"}')


@stage('import-labels', 'Validate exported labels and write canonical annotations (re-runnable)')
def cmd_import_labels(args):
    from .annotate import import_labels
    from .io import write_jsonl
    records = import_labels(args.labels, args.key, _fine_tuned_sampled(args.sample))
    out = Path(args.out)
    if out.exists():
        out.unlink()
    write_jsonl(out, records)
    print(f'{len(records)} annotations validated -> {out}')


EXTENSIONS.append(_annotation_parsers)


JUDGE_MANIFEST = V3 / 'judge_manifest.json'


def _read_manifest():
    return json.loads(JUDGE_MANIFEST.read_text()) if JUDGE_MANIFEST.exists() else {}


def _probe_parser(subs):
    pr = subs.add_parser('probe', help=STAGES['probe'][1])
    pr.add_argument('--exe', default='claude')


@stage('probe', 'Isolation probe for the headless Claude Code judge (one canary call)')
def cmd_probe(args):
    from .claude_code_backend import isolation_probe
    result = isolation_probe(args.exe)
    manifest = _read_manifest()
    manifest['probe'] = result
    dump(JUDGE_MANIFEST, manifest)
    print(json.dumps({k: result[k] for k in ('passed', 'failures', 'cli_version')}, indent=1))


EXTENSIONS.append(_probe_parser)


JUDGE_LEDGER = V3 / 'judge_ledger.json'
CF_RUNS = ('cf-biobart', 'cf-bart-lora')
# v2 packet = v1 minus 3 pronoun swaps whose gold was unsound; surviving cases are byte-identical,
# so v1 generations are reused, filtered to this packet by case ID and dialogue hash.
CF_PACKET = V3 / 'cf_cases_v2.jsonl'


def in_packet(cases, packet=CF_PACKET):
    allowed = {c['case_id']: c['source']['dialogue_sha256'] for c in read_jsonl(packet)}
    return [c for c in cases if allowed.get(c['case_id']) == c['source']['dialogue_sha256']]


def judge_items(split, sample_path=V3 / 'sample.json'):
    from .validity import classify_output
    if split == 'counterfactual':
        items = []
        for run in CF_RUNS:
            for c in in_packet(load_run(V3 / run)):
                if classify_output(c.get('prediction'), c['dialogue']) == 'summary':
                    items.append((run, c))
        return items
    sample = json.loads(Path(sample_path).read_text())
    by_run = {run: [c for c in load_run(LOCAL / run) if classify_output(c.get('prediction'), c['dialogue']) == 'summary']
              for run, _ in FINE_TUNED}
    if split in ('calibration', 'heldout'):
        ids = set(sample[split])
    else:  # all-real: every valid fine-tuned summary outside the annotation sample
        ids = {c['case_id'] for cs in by_run.values() for c in cs} - set(sample['calibration']) - set(sample['heldout'])
    return [(run, c) for run, cs in by_run.items() for c in cs if c['case_id'] in ids]


def _judge_parsers(subs):
    j = subs.add_parser('judge', help=STAGES['judge'][1])
    j.add_argument('--split', required=True, choices=['calibration', 'heldout', 'all-real', 'counterfactual'])
    j.add_argument('--trials', type=int, default=None, help='default: 2 for heldout, else 1')
    j.add_argument('--exe', default='claude')
    subs.add_parser('judge-freeze', help=STAGES['judge-freeze'][1])
    subs.add_parser('judge-unblock', help=STAGES['judge-unblock'][1])
    subs.add_parser('judge-secondary', help=STAGES['judge-secondary'][1])


@stage('judge', 'Run the isolated Claude Sonnet 5.5 judge on one split (LIVE; uses the subscription)')
def cmd_judge(args):
    import random
    from functools import partial
    from .claude_code_backend import CALL_CAP, CallLedger, call, require_run_preconditions
    from .judge_run import opaque_id, run_judge
    require_run_preconditions(_read_manifest(), args.exe)
    trials = args.trials or (2 if args.split == 'heldout' else 1)
    ledger = CallLedger(JUDGE_LEDGER, cap=CALL_CAP)
    items = judge_items(args.split)
    pending = lambda its: sum(not ledger.done(f'{args.split}:{opaque_id(c["case_id"], r)}:t{t}')
                              for r, c in its for t in range(1, trials + 1))
    room = CALL_CAP - ledger.calls_used()
    if pending(items) > room:
        if args.split != 'counterfactual':
            raise SystemExit(f'{pending(items)} calls needed but only {room} remain under the {CALL_CAP}-call cap')
        random.Random(20261109).shuffle(items)
        while items and pending(items) > room:
            items.pop()
        print(f'counterfactual reduced to {len(items)} summaries to fit the call cap')
    caller = partial(lambda payload, job, exe: call(payload, exe=exe, ledger=ledger, job_id=job), exe=args.exe)
    summary = run_judge(items, V3 / 'judge' / args.split, split=args.split, trials=trials, caller=caller,
                        ledger=ledger, freeze_manifest=JUDGE_MANIFEST)
    print(summary)


@stage('judge-freeze', 'Freeze rubric/schema/model/CLI flags after calibration (before any held-out call)')
def cmd_judge_freeze(args):
    from .judge_run import freeze_judge
    print(freeze_judge(JUDGE_MANIFEST))


@stage('judge-unblock', 'Clear a usage-limit/auth block so never-judged jobs can be re-sent')
def cmd_judge_unblock(args):
    from .claude_code_backend import CallLedger
    ledger = CallLedger(JUDGE_LEDGER)
    print('was blocked:', ledger.read().get('blocked_reason'))
    ledger.unblock()


@stage('judge-secondary', 'Free Nemotron agreement judge on held-out (LIVE; zero-price guards)')
def cmd_judge_secondary(args):
    from .judge_run import _check_freeze, run_secondary
    _check_freeze(JUDGE_MANIFEST)
    print(run_secondary(judge_items('heldout'), V3 / 'judge' / 'secondary', V3 / 'secondary_ledger.json'))


EXTENSIONS.append(_judge_parsers)


def _score_parser(subs):
    sc = subs.add_parser('score', help=STAGES['score'][1])
    sc.add_argument('--out', required=True, help='public metrics path, e.g. results/factuality/v3-<date>/metrics.json')
    sc.add_argument('--annotations', default=str(V3 / 'annotations.jsonl'))
    sc.add_argument('--sample', default=str(V3 / 'sample.json'))


@stage('score', 'Score rules and judges against human labels; write public metrics (counts and rates only)')
def cmd_score(args):
    from .checks_v3 import suggestions_v3
    from .counterfactual import score_counterfactual
    from .judge_run import load_judge
    from .v3_scoring import assemble_metrics, judge_on_counterfactual
    annotations = read_jsonl(args.annotations)
    sample = json.loads(Path(args.sample).read_text())
    cases = {(run, c['case_id']): c for run, _ in FINE_TUNED for c in load_run(LOCAL / run)}
    rules = {(a['run'], a['case_id']): {'claims': suggestions_v3(cases[(a['run'], a['case_id'])])} for a in annotations}
    claude, trials = {}, {'t1': {}, 't2': {}}
    for split in ('calibration', 'heldout'):
        merged, by_trial = load_judge(V3 / 'judge' / split)
        claude.update(merged)
        if split == 'heldout':
            trials = {k: by_trial.get(k, {}) for k in ('t1', 't2')}
    secondary = load_judge(V3 / 'judge' / 'secondary')[0] if (V3 / 'judge' / 'secondary').exists() else None
    cf_items = judge_items('counterfactual') if all((V3 / r).exists() for r in CF_RUNS) else []
    counterfactual = None
    if cf_items:
        counterfactual = {'rules_view': {r: score_counterfactual([c for run, c in cf_items if run == r]) for r in CF_RUNS},
                          'judge_on_construction_gold': judge_on_counterfactual(cf_items, load_judge(V3 / 'judge' / 'counterfactual')[0])}
    validity = json.loads((V3 / 'validity.json').read_text()) if (V3 / 'validity.json').exists() else None
    validity = {k: v['counts'] for k, v in (validity or {}).items()}
    metrics = assemble_metrics(annotations, sample, {'rules': rules, 'claude': claude}, trials=trials,
                               secondary=secondary, counterfactual=counterfactual, validity=validity)
    dump(args.out, metrics)
    for name, lines in metrics['heldout']['headline'].items():
        print(name)
        for t, text in lines.items():
            print('  ', t, '->', text)
    print(metrics['heldout']['caveat'])


EXTENSIONS.append(_score_parser)


def main(argv=None):
    args = build_parser().parse_args(argv)
    STAGES[args.stage][0](args)


if __name__ == '__main__':
    main()
