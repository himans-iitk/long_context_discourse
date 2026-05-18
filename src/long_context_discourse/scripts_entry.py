"""Console-script entry points.

Each ``run_expN`` and ``analyze_expN`` function is a thin wrapper around the
module-level ``run`` / ``analyze`` defined in :mod:`experiments`. They share
the same argparse-based CLI:

    python -m long_context_discourse.scripts_entry run_exp1 --config configs/exp1.yaml

We keep one Python entry point per command (rather than a dispatcher) so
``pyproject.toml`` console_scripts can map cleanly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from .config import Config, load_config
from .experiments import (
    exp1_relations,
    exp2_presupposition,
    exp2b_marker_ablation,
    exp3_probing,
    exp4_crosslingual,
    exp5_reasoning,
)
from .io_utils import write_json
from .logging_utils import configure_logging, get_logger

_log = get_logger(__name__)


def _build_parser(name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=name)
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--env", default=None, help="Path to a .env file (optional)")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    return parser


def _parse(name: str, argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    parser = _build_parser(name)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    config = load_config(args.config)
    return config, args


def _run(stage: str, fn: Callable[..., object], argv: list[str] | None = None) -> int:
    try:
        config, args = _parse(stage, argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    try:
        if "env_path" in fn.__code__.co_varnames:
            fn(config, env_path=args.env)
        else:
            fn(config)
    except Exception as exc:
        _log.exception("%s failed: %s", stage, exc)
        return 1
    return 0


def run_exp1(argv: list[str] | None = None) -> int:
    return _run("run_exp1", exp1_relations.run, argv)


def analyze_exp1(argv: list[str] | None = None) -> int:
    return _run("analyze_exp1", exp1_relations.analyze, argv)


def run_exp2(argv: list[str] | None = None) -> int:
    return _run("run_exp2", exp2_presupposition.run, argv)


def run_exp2b(argv: list[str] | None = None) -> int:
    return _run("run_exp2b", exp2b_marker_ablation.run, argv)


def analyze_exp2(argv: list[str] | None = None) -> int:
    rc = _run("analyze_exp2", exp2_presupposition.analyze, argv)
    if rc != 0:
        return rc
    # The 2B summary is best-effort: only run if the file exists.
    try:
        config, _ = _parse("analyze_exp2", argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if (config.paths.results_for("exp2") / "exp2b_ablation.json").is_file():
        try:
            exp2b_marker_ablation.analyze(config)
        except Exception:
            _log.exception("exp2b analysis failed; continuing")
    return 0


def run_exp3(argv: list[str] | None = None) -> int:
    return _run("run_exp3", exp3_probing.run, argv)


def analyze_exp3(argv: list[str] | None = None) -> int:
    return _run("analyze_exp3", exp3_probing.analyze, argv)


def run_exp4(argv: list[str] | None = None) -> int:
    return _run("run_exp4", exp4_crosslingual.run, argv)


def analyze_exp4(argv: list[str] | None = None) -> int:
    return _run("analyze_exp4", exp4_crosslingual.analyze, argv)


def run_exp5(argv: list[str] | None = None) -> int:
    return _run("run_exp5", exp5_reasoning.run, argv)


def analyze_exp5(argv: list[str] | None = None) -> int:
    return _run("analyze_exp5", exp5_reasoning.analyze, argv)


def compile_master_results(argv: list[str] | None = None) -> int:
    """Aggregate the per-experiment summary JSONs into one master file."""
    parser = argparse.ArgumentParser(prog="compile-master-results")
    parser.add_argument("--config", required=True, help="Any experiment YAML (used for paths)")
    parser.add_argument("--output", default=None, help="Where to write MASTER_RESULTS_FOR_PAPER.json")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    config = load_config(args.config)
    base = config.paths.results_dir

    aliases = (config.raw.get("compile") or {}).get("result_subdirs") or {}

    def _dir(key: str, default: str) -> str:
        return str(aliases.get(key, default))

    def _maybe(path: Path) -> object | None:
        if not path.is_file():
            return None
        from .io_utils import read_json

        return read_json(path)

    payload = {
        "experiment_1": _maybe(base / _dir("exp1", "exp1") / "exp1_summary.json"),
        "experiment_2": _maybe(base / _dir("exp2", "exp2") / "exp2_summary.json"),
        "experiment_2b": _maybe(base / _dir("exp2", "exp2") / "exp2b_rescue.json"),
        "experiment_3": _maybe(base / _dir("exp3", "exp3") / "exp3_analysis.json"),
        "experiment_4": _maybe(base / _dir("exp4", "exp4") / "exp4_summary_f1.json"),
        "experiment_5": _maybe(base / _dir("exp5", "exp5") / "exp5_summary.json"),
        "metadata": {
            "models": dict(config.models),
            "long_context_models": list(config.long_context_models),
            "reasoning_models": list(config.reasoning_models),
            "seed": config.seed,
            "result_subdirs": aliases or None,
        },
    }
    out = Path(args.output or (base / "MASTER_RESULTS_FOR_PAPER.json"))
    write_json(payload, out)
    _log.info("Wrote master results → %s", out)
    return 0


# Allow `python -m long_context_discourse.scripts_entry <command> ...`
def _module_main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m long_context_discourse.scripts_entry <command> ...", file=sys.stderr)
        return 2
    command = sys.argv[1].replace("-", "_")
    rest = sys.argv[2:]
    fn = globals().get(command)
    if not callable(fn):
        print(f"Unknown command {command!r}", file=sys.stderr)
        return 2
    return int(fn(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(_module_main())
