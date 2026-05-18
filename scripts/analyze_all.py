#!/usr/bin/env python3
"""Run every analyze_* in order (skipping any whose run output is missing)."""

from __future__ import annotations

import argparse
import sys

from long_context_discourse.config import load_config
from long_context_discourse.experiments import (
    exp1_relations,
    exp2_presupposition,
    exp2b_marker_ablation,
    exp3_probing,
    exp4_crosslingual,
    exp5_reasoning,
)
from long_context_discourse.logging_utils import configure_logging, get_logger

_log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Any experiment YAML for paths/models")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    config = load_config(args.config)

    pipeline = [
        ("exp1", exp1_relations.analyze, config.paths.results_for("exp1") / "exp1_rows.json"),
        ("exp2", exp2_presupposition.analyze, config.paths.results_for("exp2") / "exp2_rows.json"),
        ("exp2b", exp2b_marker_ablation.analyze, config.paths.results_for("exp2") / "exp2b_ablation.json"),
        ("exp3", exp3_probing.analyze, config.paths.results_for("exp3") / "exp3_layer_probe.json"),
        ("exp4", exp4_crosslingual.analyze, config.paths.results_for("exp4") / "exp4_rows.json"),
        ("exp5", exp5_reasoning.analyze, config.paths.results_for("exp5") / "exp5_cot_rows.json"),
    ]
    for name, fn, sentinel in pipeline:
        if not sentinel.is_file():
            _log.warning("skipping %s: %s missing", name, sentinel)
            continue
        try:
            fn(config)
        except Exception:
            _log.exception("analysis for %s failed", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
