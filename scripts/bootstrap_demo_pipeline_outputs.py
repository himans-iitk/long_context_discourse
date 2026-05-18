#!/usr/bin/env python3
"""Remove stale ``results/*`` experiment folders and emit synthetic demo JSON + PNGs.

Writes canonical paths: ``results/exp{1,2,4,5}/``, ``results/MASTER_DEMO.json``,
``figures/*_demo.png``. Safe to re-run (idempotent layout).

Usage:
  python scripts/bootstrap_demo_pipeline_outputs.py --project-root .
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
_SRC = ROOT_DIR / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _clean_results(results_dir: Path) -> None:
    if not results_dir.is_dir():
        results_dir.mkdir(parents=True, exist_ok=True)
        return
    for child in list(results_dir.iterdir()):
        if child.name in {".gitkeep"}:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap synthetic demo pipeline outputs.")
    ap.add_argument("--project-root", type=Path, default=ROOT_DIR)
    args = ap.parse_args()
    root = args.project_root.resolve()

    from long_context_discourse.demo_pipeline_outputs import emit_demo_project

    _clean_results(root / "results")
    emit_demo_project(root)
    print(f"Demo artefacts written under {root / 'results'} and {root / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
