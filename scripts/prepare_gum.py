#!/usr/bin/env python3
"""CLI wrapper around :mod:`long_context_discourse.preprocess.gum`.

Examples
--------
Smoke run (5 files):

    python scripts/prepare_gum.py \\
        --rs4-dir ../../long_context_discourse_dataset/GUM/rst/rstweb \\
        --out-path ../../long_context_discourse_dataset/processed/exp3/gum_rst_processed.json \\
        --smoke 5

Full run: drop ``--smoke``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from long_context_discourse.logging_utils import configure_logging
from long_context_discourse.preprocess.gum import build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rs4-dir", required=True, type=Path)
    parser.add_argument("--out-path", required=True, type=Path)
    parser.add_argument(
        "--smoke",
        type=int,
        default=None,
        metavar="N",
        help="Smoke mode: only the first N .rs4 files are processed",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    out = build(rs4_dir=args.rs4_dir, out_path=args.out_path, file_limit=args.smoke)
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
