#!/usr/bin/env python3
"""CLI wrapper for :mod:`long_context_discourse.merge_parallel`."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from long_context_discourse.merge_parallel import main_merge

if __name__ == "__main__":
    raise SystemExit(main_merge())
