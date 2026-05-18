#!/usr/bin/env python3
"""CLI wrapper around :mod:`long_context_discourse.preprocess.ted_mdb`.

Examples
--------
Smoke run (1 file per language):

    python scripts/prepare_ted_mdb.py \\
        --ted-root ../../long_context_discourse_dataset/Ted-MDB \\
        --out-path ../../long_context_discourse_dataset/processed/exp4/ted_mdb_processed.json \\
        --smoke 1

Full run: drop ``--smoke``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from long_context_discourse.logging_utils import configure_logging
from long_context_discourse.preprocess.ted_mdb import DEFAULT_LANGUAGES, build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ted-root", required=True, type=Path)
    parser.add_argument("--out-path", required=True, type=Path)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
    )
    parser.add_argument(
        "--smoke",
        type=int,
        default=None,
        metavar="N",
        help="Smoke mode: only the first N annotation files per language",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    out = build(
        ted_root=args.ted_root,
        out_path=args.out_path,
        languages=args.languages,
        file_limit_per_lang=args.smoke,
    )
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
