#!/usr/bin/env python3
"""Emit per-model YAML shards for parallel OpenRouter runs (merge step follows).

Shards extend the production configs under configs/ and override ``experiment``
to ``expN__shard_<model>`` so outputs land in disjoint ``results/`` folders.

- Exp 1 / 2 / 2B shards: **every** model in ``configs/default.yaml``.
- Exp 4 shards: ``models_subset`` from ``configs/exp4.yaml`` only.
- Exp 5 shards: ``cot_models`` from ``configs/exp5.yaml`` only.

Usage:
  python scripts/generate_parallel_shard_configs.py \\
    --project-root . --output-dir configs/parallel/shards
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from long_context_discourse.config import ConfigError, load_config

_MODEL_ORDER: tuple[str, ...] = (
    "gpt4o",
    "deepseek_r1",
    "llama3_70b",
    "llama3_8b",
    "mistral_7b",
    "mixtral",
    "phi4",
    "deepseek_v3",
)


def _load_models_from_default(project_root: Path) -> tuple[str, ...]:
    default_path = project_root / "configs" / "default.yaml"
    if not default_path.is_file():
        return _MODEL_ORDER
    raw = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
    reg = raw.get("models") or {}
    if not isinstance(reg, dict) or not reg:
        return _MODEL_ORDER
    keys = list(reg.keys())
    ordered = [m for m in _MODEL_ORDER if m in keys]
    extras = [m for m in keys if m not in ordered]
    return tuple(ordered + extras)


def _exp4_shard_models(project_root: Path) -> tuple[str, ...]:
    cfg = load_config(project_root / "configs" / "exp4.yaml")
    raw = cfg.raw.get("models_subset")
    if not raw:
        raise ConfigError("configs/exp4.yaml must define models_subset for parallel Exp 4 shards")
    return tuple(str(m) for m in raw)


def _exp5_shard_models(project_root: Path) -> tuple[str, ...]:
    cfg = load_config(project_root / "configs" / "exp5.yaml")
    cot = cfg.section("run").get("cot_models")
    if not cot:
        raise ConfigError("configs/exp5.yaml must define run.cot_models for parallel Exp 5 shards")
    return tuple(str(m) for m in cot)


def _write(path: Path, body: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(body, f, sort_keys=False, allow_unicode=True)


def _rel_yaml(base: str) -> str:
    return str(Path("..", "..", base).as_posix())


def generate(
    project_root: Path,
    output_dir: Path,
    all_models: tuple[str, ...],
    exp4_models: tuple[str, ...],
    exp5_models: tuple[str, ...],
) -> list[Path]:
    written: list[Path] = []
    out_dir = output_dir.resolve()

    for m in all_models:
        exp_tag = f"__shard_{m}"
        _write(
            out_dir / f"exp1.shard_{m}.yaml",
            {
                "extends": _rel_yaml("exp1.yaml"),
                "experiment": f"exp1{exp_tag}",
                "run": {"models_subset": [m]},
            },
        )
        written.append(out_dir / f"exp1.shard_{m}.yaml")

        _write(
            out_dir / f"exp2.shard_{m}.yaml",
            {
                "extends": _rel_yaml("exp2.yaml"),
                "experiment": f"exp2{exp_tag}",
                "run": {"models_subset": [m]},
                "ablation": {"models": [m]},
            },
        )
        written.append(out_dir / f"exp2.shard_{m}.yaml")

    for m in exp4_models:
        exp_tag = f"__shard_{m}"
        _write(
            out_dir / f"exp4.shard_{m}.yaml",
            {
                "extends": _rel_yaml("exp4.yaml"),
                "experiment": f"exp4{exp_tag}",
                "models_subset": [m],
            },
        )
        written.append(out_dir / f"exp4.shard_{m}.yaml")

    for m in exp5_models:
        exp_tag = f"__shard_{m}"
        _write(
            out_dir / f"exp5.shard_{m}.yaml",
            {
                "extends": _rel_yaml("exp5.yaml"),
                "experiment": f"exp5{exp_tag}",
                "run": {"cot_models": [m]},
            },
        )
        written.append(out_dir / f"exp5.shard_{m}.yaml")

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate parallel shard YAML configs.")
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--output-dir", type=Path, default=Path("configs/parallel/shards"))
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Override all-models list for Exp 1/2 only; default from configs/default.yaml",
    )
    args = ap.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir
    if not out.is_absolute():
        out = (root / out).resolve()

    all_models = tuple(args.models) if args.models else _load_models_from_default(root)
    exp4_models = _exp4_shard_models(root)
    exp5_models = _exp5_shard_models(root)

    for stale in out.glob("exp*.shard_*.yaml"):
        stale.unlink()

    paths = generate(root, out, all_models, exp4_models, exp5_models)
    print(f"Wrote {len(paths)} shard configs under {out}")


if __name__ == "__main__":
    main()
