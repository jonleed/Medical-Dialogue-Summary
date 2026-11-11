"""Export the factuality benchmark: everything needed to rebuild the published metrics offline,
and to score another summarizer against the same dialogues, plans and labels.

Run names become stable model names, judge opaque IDs are resolved, and run-time bookkeeping
(call ledgers, raw duplicate responses, token IDs, account confirmations) is left out.
"""
import json
import shutil
from pathlib import Path

from .io import read_jsonl, write_jsonl
from .sections import canonical_section

FILES = ('DATASHEET.md', 'cases.jsonl', 'summaries.jsonl', 'counterfactual_cases.jsonl', 'counterfactual_summaries.jsonl',
         'plans.jsonl', 'labels.jsonl', 'splits.json', 'judge_outputs.jsonl', 'judge_config.json')
OPTIONAL = {'rubric_log.md': 'rubric_revisions.md'}
_INPUT_FIELDS = ('text', 'truncated', 'full_token_count', 'sha256')
_PROVENANCE_FIELDS = ('base_revision', 'adapter_weights_sha256')


def _summary(model, case):
    mi = case.get('model_input') or {}
    prov = case.get('provenance') or {}
    return {'model': model, 'case_id': case['case_id'], 'prediction': case.get('prediction'),
            'model_input': {k: mi[k] for k in _INPUT_FIELDS if k in mi},
            'provenance': {k: prov[k] for k in _PROVENANCE_FIELDS if k in prov}, 'protocol': case.get('protocol')}


def _case(case):
    out = {k: case[k] for k in ('case_id', 'source', 'dialogue', 'requested_section', 'reference') if k in case}
    out['canonical_section'] = canonical_section(case)
    if 'counterfactual' in case:
        out['counterfactual'] = case['counterfactual']
    return out


def _judge_rows(judge_root, models):
    rows = []
    for split_dir in sorted(p for p in Path(judge_root).iterdir() if p.is_dir() and not p.name.startswith('secondary')):
        key = json.loads((split_dir / 'key.json').read_text()) if (split_dir / 'key.json').exists() else {}
        for oid, ident in sorted(key.items()):
            for path in sorted((split_dir / oid).glob('t*.json')):
                r = json.loads(path.read_text())
                row = {'model': models[ident['run']], 'case_id': ident['case_id'], 'split': split_dir.name,
                       'trial': r.get('trial'), 'status': r.get('status')}
                for k in ('failure_reason', 'result', 'cli_version', 'rubric_hash', 'schema_hash', 'usage'):
                    if k in r:
                        row[k] = r[k]
                rows.append(row)
    return rows


def _datasheet(n_cases, models, n_labels):
    return f'''# Clinical summary factuality benchmark

Data for rebuilding this repository's factuality results and for scoring other summarizers on the same task.

## Contents

| File | What it holds |
|---|---|
| `cases.jsonl` | {n_cases} MTS-Dialog validation dialogues with the requested note section and the reference note |
| `summaries.jsonl` | Summaries from {", ".join(models)}, with the text each model could see and model provenance |
| `counterfactual_cases.jsonl` | Edited dialogues (age, pronoun, race), each with its exact edits and the expected answer |
| `counterfactual_summaries.jsonl` | Summaries of the edited dialogues |
| `plans.jsonl` | Expected facts for each sampled dialogue's section, drafted from the dialogue alone before any summary was shown |
| `labels.jsonl` | {n_labels} reviewed claim-level labels (supported, contradicted, unsupported, unclear) and present/omitted judgments for every planned fact |
| `splits.json` | Calibration / held-out assignment and how each dialogue was sampled |
| `judge_outputs.jsonl` | The LLM judge's validated output for every summary it reviewed, by split and trial |
| `judge_config.json` | Judge model, Claude Code version, flags, and rubric/schema hashes |

Spans are 0-based, end-exclusive character offsets into the text they name. The judge rubric is `factuality/prompts/judge.txt`, and the annotation rules are in `docs/factuality_annotation_guide.md`.

## Labels

Plans and claim labels were drafted by Claude Sonnet in a separate, isolated session. The drafter worked from the labeling sheets and had no other context. For plans, the sheet was source-only: each dialogue and its requested section, with plans frozen before any summary was shown. For labels, the sheet showed the summaries in shuffled order with no model identity, rule-checker output or judge output. The drafter listed the facts each section should contain, labeled every atomic claim as supported, contradicted, unsupported or unclear, and marked each planned fact as present or omitted. Jonathan Leeder, a non-clinician, reviewed every item and is the reviewer of record (named in each record). The labels are not a clinical gold standard. There is a single reviewer and no independent second label. The drafter and the LLM judge are both Claude models.

## License and attribution

Dialogues, reference notes and quoted spans come from MTS-Dialog (Ben Abacha et al., *An Empirical Study of Clinical Note Generation from Doctor-Patient Encounters*, EACL 2023; https://github.com/abachaa/MTS-Dialog), licensed under CC BY 4.0. Summaries, plans, labels and judge outputs are derived additions, released under the same license. The dialogues are a research dataset, not real patient records. Nothing here is medical advice.
'''


def export_bundle(local_root, out_dir, runs, cf_runs, v3_dir=None, packet='cf_cases_v2.jsonl'):
    """runs/cf_runs: {local run folder name: published model name}."""
    local, out = Path(local_root), Path(out_dir)
    v3 = Path(v3_dir) if v3_dir else local / 'v3'
    if out.exists():
        raise ValueError(f'{out} exists; export into a new directory')
    out.mkdir(parents=True)
    by_run = {run: read_jsonl(local / run / 'cases.jsonl') for run in runs}
    first = next(iter(by_run.values()))
    write_jsonl(out / 'cases.jsonl', [_case(c) for c in sorted(first, key=lambda c: c['case_id'])])
    write_jsonl(out / 'summaries.jsonl', [_summary(runs[run], c) for run, cases in by_run.items()
                                          for c in sorted(cases, key=lambda c: c['case_id'])])
    packet_cases = read_jsonl(v3 / packet)
    allowed = {c['case_id'] for c in packet_cases}
    write_jsonl(out / 'counterfactual_cases.jsonl', [_case(c) for c in packet_cases])
    write_jsonl(out / 'counterfactual_summaries.jsonl', [_summary(model, c) for run, model in cf_runs.items()
                                                         for c in read_jsonl(v3 / run / 'cases.jsonl') if c['case_id'] in allowed])
    shutil.copyfile(v3 / 'plans' / 'plans.jsonl', out / 'plans.jsonl')
    labels = []
    for a in read_jsonl(v3 / 'annotations.jsonl'):
        a = dict(a, model=runs[a['run']])
        a.pop('run')
        labels.append(a)
    write_jsonl(out / 'labels.jsonl', labels)
    shutil.copyfile(v3 / 'sample.json', out / 'splits.json')
    write_jsonl(out / 'judge_outputs.jsonl', _judge_rows(v3 / 'judge', {**runs, **cf_runs}))
    manifest = json.loads((v3 / 'judge_manifest.json').read_text())
    config = {k: v for k, v in manifest.items() if not k.startswith('terms_')}
    (out / 'judge_config.json').write_text(json.dumps(config, indent=2) + '\n')
    for src, dst in OPTIONAL.items():
        if (v3 / src).exists():
            shutil.copyfile(v3 / src, out / dst)
    (out / 'DATASHEET.md').write_text(_datasheet(len(first), list(runs.values()), len(labels)))
    return out


def export_from_config(config, out_dir):
    """Export using the saved runs declared in factuality.toml (run_dir / counterfactual_run_dir)."""
    runs = {m.run_dir: name for name, m in config.models.items() if m.run_dir}
    cf_runs = {m.counterfactual_run_dir: name for name, m in config.models.items() if m.counterfactual_run_dir}
    return export_bundle(config.work, out_dir, runs, cf_runs, v3_dir=config.study,
                         packet=config.counterfactual_packet.name)
