"""YAML-backed experiment configuration.

The loader supports a single ``extends:`` pointer to another YAML file. The
parent config is loaded recursively and the child is deep-merged on top, so
``configs/exp1.yaml`` only has to specify what differs from
``configs/default.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or contradictory."""


@dataclass(frozen=True)
class PathsConfig:
    dataset_root: Path
    results_dir: Path
    figures_dir: Path
    checkpoints_subdir: str = "checkpoints"

    def results_for(self, experiment: str) -> Path:
        return self.results_dir / experiment

    def checkpoints_for(self, experiment: str) -> Path:
        return self.results_for(experiment) / self.checkpoints_subdir


@dataclass(frozen=True)
class OpenRouterConfig:
    base_url: str
    api_key_env: str
    referer_env: str
    title_env: str
    request_timeout_seconds: float
    max_retries: int
    retry_base_seconds: float
    rate_limit_sleep_seconds: float


@dataclass(frozen=True)
class Config:
    """In-memory, validated experiment config."""

    experiment: str
    description: str
    paths: PathsConfig
    openrouter: OpenRouterConfig
    models: Mapping[str, str]
    long_context_models: tuple[str, ...]
    reasoning_models: tuple[str, ...]
    seed: int
    temperature: float
    raw: Mapping[str, Any] = field(repr=False)

    def section(self, name: str) -> Mapping[str, Any]:
        """Return a sub-section of the original YAML (e.g. ``run``, ``data``)."""
        value = self.raw.get(name, {})
        if not isinstance(value, Mapping):
            raise ConfigError(f"Section {name!r} must be a mapping; got {type(value).__name__}")
        return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``override`` deep-merged on top of ``base``."""
    out: dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _load_yaml_with_extends(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    if _seen is None:
        _seen = set()
    if path in _seen:
        raise ConfigError(f"Cyclic 'extends' chain at {path}")
    _seen.add(path)

    with path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must define a mapping at the top level")

    parent_ref = loaded.pop("extends", None)
    if parent_ref:
        parent_path = (path.parent / parent_ref).resolve()
        parent_data = _load_yaml_with_extends(parent_path, _seen)
        return _deep_merge(parent_data, loaded)
    return loaded


def _infer_project_root(config_path: Path) -> Path:
    """Resolve the repo root for relative ``paths.*`` in YAML.

    Walk upward from the config's directory for ``pyproject.toml`` so nested
    configs (e.g. ``configs/parallel/shards/*.yaml``) still resolve correctly.
    If none is found (minimal test trees), fall back to ``config.parent.parent``
    as when configs live directly under ``configs/``.
    """
    cur = config_path.parent
    while True:
        if (cur / "pyproject.toml").is_file():
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return config_path.parent.parent


def load_config(config_path: str | Path, project_root: Path | None = None) -> Config:
    """Load and validate a YAML config from disk.

    ``project_root`` is used to resolve relative ``paths.*`` entries; by default
    it is inferred from ``pyproject.toml``, or ``config.parent.parent`` if that
    file is absent (e.g. tiny fixture trees).
    """
    config_path = Path(config_path).expanduser().resolve()
    raw = _load_yaml_with_extends(config_path)

    project_root = (project_root or _infer_project_root(config_path)).resolve()

    paths_block = raw.get("paths", {})
    paths = PathsConfig(
        dataset_root=_resolve(project_root, paths_block.get("dataset_root", "data")),
        results_dir=_resolve(project_root, paths_block.get("results_dir", "results")),
        figures_dir=_resolve(project_root, paths_block.get("figures_dir", "figures")),
        checkpoints_subdir=paths_block.get("checkpoints_subdir", "checkpoints"),
    )

    or_block = raw.get("openrouter", {})
    openrouter = OpenRouterConfig(
        base_url=or_block.get("base_url", "https://openrouter.ai/api/v1"),
        api_key_env=or_block.get("api_key_env", "OPENROUTER_API_KEY"),
        referer_env=or_block.get("referer_env", "OPENROUTER_REFERER"),
        title_env=or_block.get("title_env", "OPENROUTER_TITLE"),
        request_timeout_seconds=float(or_block.get("request_timeout_seconds", 120)),
        max_retries=int(or_block.get("max_retries", 4)),
        retry_base_seconds=float(or_block.get("retry_base_seconds", 1.5)),
        rate_limit_sleep_seconds=float(or_block.get("rate_limit_sleep_seconds", 0.15)),
    )

    models = raw.get("models", {})
    if not isinstance(models, Mapping) or not models:
        raise ConfigError("'models' must be a non-empty mapping of short_name → openrouter_id")

    return Config(
        experiment=str(raw.get("experiment") or config_path.stem),
        description=str(raw.get("description", "")),
        paths=paths,
        openrouter=openrouter,
        models=dict(models),
        long_context_models=tuple(raw.get("long_context_models", [])),
        reasoning_models=tuple(raw.get("reasoning_models", [])),
        seed=int(raw.get("seed", 42)),
        temperature=float(raw.get("temperature", 0.0)),
        raw=raw,
    )


def _resolve(project_root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (project_root / candidate).resolve()
    return candidate
