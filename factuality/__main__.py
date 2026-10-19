"""Run `python -m factuality --help` from the repository root."""
import argparse
from pathlib import Path

from .io import read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='stage', required=True)
    export = subs.add_parser('export', help='Export deterministic validation sample, no model load')
    export.add_argument('--n',type=int,default=100)
    export.add_argument('--seed',type=int,default=42)
    export.add_argument('--out',required=True)
    generate = subs.add_parser('generate',help='Generate via unmodified Flask route; use offline HF env for cached-only execution')
    generate.add_argument('--cases',required=True)
    generate.add_argument('--out',required=True)
    generate.add_argument('--threads',type=int,default=4)
    generate.add_argument('--base-model',help='Override serving BASE_MODEL_ID for a paired baseline')
    generate.add_argument('--adapter',help='Local adapter directory, or none for base-only loader substitution')
    plan = subs.add_parser('plan',help='Blind source-only expected-fact planning; predictions omitted')
    plan.add_argument('--cases',required=True)
    plan.add_argument('--out',required=True)
    queue = subs.add_parser('queue',help='Pending JSONL, CSV and local HTML review queue')
    queue.add_argument('--cases',required=True)
    queue.add_argument('--plans',help='Optional completed source-only JSONL plans')
    queue.add_argument('--out',required=True)
    score = subs.add_parser('score',help='Validate and score annotations by reviewer kind; no regex gold labels')
    score.add_argument('--cases',required=True)
    score.add_argument('--annotations')
    score.add_argument('--out',required=True)
    audit = subs.add_parser('audit-cache',help='Audit historical predictions; reference matches never prove generation identity')
    audit.add_argument('--cache',required=True)
    audit.add_argument('--cases',required=True)
    audit.add_argument('--out',required=True)
    rouge = subs.add_parser('rouge',help='Separate served-output ROUGE, no downloads')
    rouge.add_argument('--cases',required=True)
    rouge.add_argument('--out',required=True)
    paired = subs.add_parser('paired',help='Check exact paired cases/settings and produce separate overlap comparisons')
    paired.add_argument('--runs',nargs='+',required=True)
    paired.add_argument('--out',required=True)
    summarize = subs.add_parser('summarize',help='Finalize a complete local run into metrics, manifest, and pending review queue')
    summarize.add_argument('--run',required=True)
    summarize.add_argument('--out',required=True)
    summarize.add_argument('--annotations')
    summarize.add_argument('--seed',type=int,default=42)
    check = subs.add_parser('check',help='Versioned demographic suggestions; optional selected-review development regressions')
    check.add_argument('--cases',required=True)
    check.add_argument('--annotations')
    check.add_argument('--out',required=True)
    freeze = subs.add_parser('challenge-freeze',help='Freeze controlled synthetic capability cases before scoring')
    freeze.add_argument('--out',required=True)
    challenge = subs.add_parser('challenge-run',help='Measure the frozen checker; retain misses in denominators')
    challenge.add_argument('--frozen',required=True)
    challenge.add_argument('--out',required=True)
    challenge.add_argument('--require-targets',action='store_true',help='Exit 2 after writing results when a declared engineering target fails')
    judge_export = subs.add_parser('judge-export',help='Blinded semantic-judge tasks; no external calls')
    judge_export.add_argument('--cases',nargs='+',required=True)
    judge_export.add_argument('--annotations',nargs='+')
    judge_export.add_argument('--challenge')
    judge_export.add_argument('--out',required=True)
    judge_score = subs.add_parser('judge-score',help='Validate actual judge responses and produce meta-evaluation/disagreement reports')
    judge_score.add_argument('--tasks',required=True)
    judge_score.add_argument('--responses',nargs='*',default=[])
    judge_score.add_argument('--reference-labels')
    judge_score.add_argument('--out',required=True)
    args = parser.parse_args()
    if args.stage=='challenge-freeze':
        from .challenge import freeze
        print(freeze(args.out))
    elif args.stage=='challenge-run':
        from .challenge import run
        result=run(args.frozen,args.out)
        print(result['engineering_targets_met'])
        if args.require_targets and not all(result['engineering_targets_met'].values()):raise SystemExit(2)
    elif args.stage=='judge-export':
        from .judging import prepare
        print(prepare(args.cases,args.out,args.annotations,args.challenge))
    elif args.stage=='judge-score':
        from .judging import judge_report
        if Path(args.out).exists():raise ValueError('Judge report already exists; use a new path')
        responses=[r for path in args.responses for r in read_jsonl(path)]
        write_json(args.out,judge_report(read_jsonl(args.tasks),responses,read_jsonl(args.reference_labels) if args.reference_labels else None))
    elif args.stage=='check':
        from .demographic_audit import audit_run
        result=audit_run(args.cases,args.out,args.annotations)
        print(result['automatic'])
    elif args.stage=='export':
        from .export import sample_cases
        write_jsonl(args.out,sample_cases(n=args.n,seed=args.seed))
    elif args.stage=='generate':
        from .generate import generate
        generate(args.cases,args.out,args.threads,args.base_model,args.adapter)
    elif args.stage=='plan':
        from .review import blind_plans
        write_jsonl(args.out,blind_plans(read_jsonl(args.cases)))
    elif args.stage=='queue':
        from .review import write_queue
        summary = write_queue(read_jsonl(args.cases),args.out,read_jsonl(args.plans) if args.plans else None)
        print(summary)
    elif args.stage=='score':
        from .scoring import score
        write_json(args.out,score(read_jsonl(args.cases),read_jsonl(args.annotations) if args.annotations else []))
    elif args.stage=='audit-cache':
        from .export import audit_cache
        write_json(args.out,audit_cache(args.cache,read_jsonl(args.cases)))
    elif args.stage=='summarize':
        from .summarize import summarize_run
        print(summarize_run(args.run,args.out,args.annotations,args.seed))
    elif args.stage=='paired':
        from .paired import compare_runs
        write_json(args.out,compare_runs(args.runs))
    elif args.stage=='rouge':
        from .rouge import compute_rouge
        write_json(args.out,compute_rouge(read_jsonl(args.cases)))


if __name__=='__main__': main()
