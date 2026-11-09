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


def main(argv=None):
    args = build_parser().parse_args(argv)
    STAGES[args.stage][0](args)


if __name__ == '__main__':
    main()
