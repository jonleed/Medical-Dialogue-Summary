"""Case export and Flask-route generation: python -m factuality {export,generate} --help.

The evaluation stages (screening, counterfactuals, annotation, judging, scoring) live in
`python -m factuality.cli`.
"""
import argparse

from .io import write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='stage', required=True)
    export = subs.add_parser('export', help='Export the deterministic validation sample (no model load)')
    export.add_argument('--n', type=int, default=100)
    export.add_argument('--seed', type=int, default=42)
    export.add_argument('--out', required=True)
    generate = subs.add_parser('generate', help='Generate through the unmodified Flask route; use offline HF env for cached-only runs')
    generate.add_argument('--cases', required=True)
    generate.add_argument('--out', required=True)
    generate.add_argument('--threads', type=int, default=4)
    generate.add_argument('--base-model', help='Override serving BASE_MODEL_ID')
    generate.add_argument('--adapter', help='Local adapter directory, or none for base-only')
    args = parser.parse_args()
    if args.stage == 'export':
        from .export import sample_cases
        write_jsonl(args.out, sample_cases(n=args.n, seed=args.seed))
    else:
        from .generate import generate
        generate(args.cases, args.out, args.threads, args.base_model, args.adapter)


if __name__ == '__main__':
    main()
