"""Load `factuality.toml`: the summarizers under evaluation, their pinned revisions, and paths."""
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = ('flask', 'hf', 'command')
_PATHS = ('benchmark', 'work', 'published', 'study', 'counterfactual_packet')


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    generator: str
    label: str
    base_model: str | None = None
    base_revision: str | None = None
    adapter: str | None = None
    adapter_revision: str | None = None
    command: list[str] | None = None
    run_dir: str | None = None                 # saved run under `work` (benchmark-building stages only)
    counterfactual_run_dir: str | None = None  # saved counterfactual run under `study`


@dataclass(frozen=True)
class Config:
    benchmark: Path
    work: Path
    published: Path
    study: Path
    counterfactual_packet: Path
    models: dict[str, ModelSpec] = field(default_factory=dict)


def _model(name, raw):
    generator = raw.get('generator')
    if generator not in GENERATORS:
        raise ConfigError(f'model {name!r}: generator must be one of {", ".join(GENERATORS)}')
    if generator in ('flask', 'hf') and not (raw.get('base_model') and raw.get('base_revision')):
        raise ConfigError(f'model {name!r}: {generator} generator needs base_model and a pinned base_revision')
    if generator == 'command' and not raw.get('command'):
        raise ConfigError(f'model {name!r}: command generator needs a non-empty command list')
    known = set(ModelSpec.__dataclass_fields__) - {'name'}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f'model {name!r}: unknown keys {sorted(unknown)}')
    return ModelSpec(name=name, label=raw.get('label', name), **{k: v for k, v in raw.items() if k != 'label'})


def load_config(path=ROOT / 'factuality.toml'):
    path = Path(path)
    data = tomllib.loads(path.read_text(encoding='utf-8'))
    paths = data.get('paths', {})
    missing = [k for k in _PATHS if k not in paths]
    if missing:
        raise ConfigError(f'[paths] is missing {missing}')
    resolved = {k: Path(os.path.normpath(path.absolute().parent / paths[k])) for k in _PATHS}
    return Config(**resolved, models={n: _model(n, raw) for n, raw in data.get('models', {}).items()})
