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
    report = {Path(r).name: score_counterfactual(read_jsonl(Path(r) / 'cases.jsonl')) for r in args.runs}
    dump(args.out, report)
    for name, r in report.items():
        print(name, {k: v for k, v in r['families'].items()})


EXTENSIONS.append(_cf_parsers)


def main(argv=None):
    args = build_parser().parse_args(argv)
    STAGES[args.stage][0](args)


if __name__ == '__main__':
    main()
