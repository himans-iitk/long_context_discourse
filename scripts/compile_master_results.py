#!/usr/bin/env python3
from __future__ import annotations

import sys

from long_context_discourse.scripts_entry import compile_master_results

if __name__ == "__main__":
    raise SystemExit(compile_master_results(sys.argv[1:]))
