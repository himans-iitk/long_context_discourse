#!/usr/bin/env python3
"""CLI wrapper around :mod:`long_context_discourse.preprocess.pdtb`.

Examples
--------
Smoke run (≤ 2 files per section, no balancing constraint relaxed):

    python scripts/prepare_pdtb.py \\
        --pdtb-root /Volumes/LDC2019T05/PDTB-3.0 \\
        --out-dir   ../../long_context_discourse_dataset/processed/pdtb \\
        --smoke 2

Full run:

    python scripts/prepare_pdtb.py \\
        --pdtb-root /Volumes/LDC2019T05/PDTB-3.0 \\
        --out-dir   ../../long_context_discourse_dataset/processed/pdtb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from long_context_discourse.logging_utils import configure_logging
from long_context_discourse.preprocess.pdtb import (
    DEFAULT_TEST_SECTIONS,
    DEFAULT_TRAIN_SECTIONS,
    build,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdtb-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--per-cell",
        type=int,
        default=50,
        help="Examples per (rel_type × sense_l1) cell in the balanced test set",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-sections",
        nargs="+",
        default=list(DEFAULT_TRAIN_SECTIONS),
        help="WSJ section ids for the padding / training pool (default: 02–21)",
    )
    parser.add_argument(
        "--test-sections",
        nargs="+",
        default=list(DEFAULT_TEST_SECTIONS),
        help="WSJ section ids for the test set (default: 23)",
    )
    parser.add_argument(
        "--smoke",
        type=int,
        default=None,
        metavar="N",
        help="Smoke mode: only the first N files per section are read",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    paths = build(
        pdtb_root=args.pdtb_root,
        out_dir=args.out_dir,
        train_sections=args.train_sections,
        test_sections=args.test_sections,
        per_cell=args.per_cell,
        seed=args.seed,
        smoke_files_per_section=args.smoke,
    )
    print("Wrote:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
