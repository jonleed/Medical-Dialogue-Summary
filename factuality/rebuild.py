"""Rebuild every published factuality aggregate offline from the benchmark bundle.

The bundle (data/factuality_benchmark/) holds the cases, summaries, plans, reference labels and judge
outputs; nothing here calls a model or the judge. Output is byte-stable for a given bundle.
"""
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .counterfactual import score_counterfactual
from .io import read_jsonl
from .judge_run import merge_trials
from .metrics import assemble_metrics, demographic_error_rates, judge_on_counterfactual, RESAMPLES
from .rules import suggestions_v3
from .screening import screen_runs
from .validity import classify_output, validity_summary

OUTPUTS = ('metrics.json', 'screening.json', 'counterfactual.json', 'demographic_error_rates.json')


@dataclass
class Bundle:
    cases: dict
    summaries: dict
    cf_cases: list
    cf_summaries: dict
    plans: list
    labels: list
    splits: dict
    judge: dict = field(default_factory=dict)
    judge_trials: dict = field(default_factory=dict)
    judge_config: dict = field(default_factory=dict)


def _join(cases_by_id, rows):
    out = defaultdict(list)
    for r in rows:
        case = dict(cases_by_id[r['case_id']])
        case.update({k: r[k] for k in ('prediction', 'model_input', 'provenance', 'protocol') if k in r})
        out[r['model']].append(case)
    return dict(out)


def load_bundle(path):
    path = Path(path)
    cases = {c['case_id']: c for c in read_jsonl(path / 'cases.jsonl')}
    cf_cases = read_jsonl(path / 'counterfactual_cases.jsonl')
    grouped = defaultdict(list)
    trials = defaultdict(dict)
    for r in read_jsonl(path / 'judge_outputs.jsonl'):
        if r['status'] == 'complete' and 'result' in r:
            grouped[(r['split'], r['model'], r['case_id'])].append((r['trial'], r['result']))
            if r['split'] == 'heldout':
                trials[f"t{r['trial']}"][(r['model'], r['case_id'])] = r['result']
    judge = defaultdict(dict)
    for (split, model, case_id), results in grouped.items():
        judge[split][(model, case_id)] = merge_trials([res for _, res in sorted(results, key=lambda t: t[0])])
    return Bundle(cases=cases, summaries=_join(cases, read_jsonl(path / 'summaries.jsonl')), cf_cases=cf_cases,
                  cf_summaries=_join({c['case_id']: c for c in cf_cases}, read_jsonl(path / 'counterfactual_summaries.jsonl')),
                  plans=read_jsonl(path / 'plans.jsonl'), labels=read_jsonl(path / 'labels.jsonl'),
                  splits=json.loads((path / 'splits.json').read_text()), judge=dict(judge), judge_trials=dict(trials),
                  judge_config=json.loads((path / 'judge_config.json').read_text()))


def _latest(judge, split):
    rounds = [s for s in judge if s == split or s.startswith(split + '-r')]
    if not rounds:
        return {}
    return judge[max(rounds, key=lambda s: int(s.rsplit('-r', 1)[1]) if '-r' in s else 1)]


def compute(bundle, resamples=RESAMPLES):
    screening = screen_runs([(model, model, cases) for model, cases in bundle.summaries.items()])
    counterfactual = {model: score_counterfactual(cases) for model, cases in bundle.cf_summaries.items()}
    annotations = [dict(a, run=a['model']) for a in bundle.labels]
    summary_cases = {(m, c['case_id']): c for m, cs in bundle.summaries.items() for c in cs}
    rules = {(a['run'], a['case_id']): {'claims': suggestions_v3(summary_cases[(a['run'], a['case_id'])])} for a in annotations}
    claude = {**_latest(bundle.judge, 'calibration'), **bundle.judge.get('heldout', {})}
    cf_items = [(m, c) for m, cs in bundle.cf_summaries.items() for c in cs
                if classify_output(c.get('prediction'), c['dialogue']) == 'summary']
    cf_block = {'rules_view': counterfactual,
                'judge_on_construction_gold': judge_on_counterfactual(cf_items, bundle.judge.get('counterfactual', {}))}
    validity = {m: validity_summary(cs)['counts'] for m, cs in bundle.summaries.items()}
    trials = {k: bundle.judge_trials.get(k, {}) for k in ('t1', 't2')}
    metrics = assemble_metrics(annotations, bundle.splits, {'rules': rules, 'claude': claude}, trials=trials,
                               counterfactual=cf_block, validity=validity, resamples=resamples)
    random_ids = set(bundle.splits['random']) | set(bundle.splits.get('top_up', []))
    return {'metrics.json': metrics, 'screening.json': screening, 'counterfactual.json': counterfactual,
            'demographic_error_rates.json': demographic_error_rates(bundle.labels, random_ids, resamples)}


def _dump(obj):
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n'


def rebuild(bundle_dir, out_dir, resamples=RESAMPLES):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, obj in compute(load_bundle(bundle_dir), resamples).items():
        (out / name).write_text(_dump(obj), encoding='utf-8')
        written[name] = out / name
    return written


def check(bundle_dir, published_dir, resamples=RESAMPLES):
    """Names of published files that differ from a fresh rebuild (empty list = reproduced exactly)."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh = rebuild(bundle_dir, tmp, resamples)
        return [n for n in OUTPUTS if not (Path(published_dir) / n).exists()
                or (Path(published_dir) / n).read_bytes() != fresh[n].read_bytes()]
