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


def main(argv=None):
    args = build_parser().parse_args(argv)
    STAGES[args.stage][0](args)


if __name__ == '__main__':
    main()
