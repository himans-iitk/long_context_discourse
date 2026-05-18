#!/usr/bin/env python3
"""Thin shell-friendly wrapper around the run_exp1 entry point."""

from __future__ import annotations

import sys

from long_context_discourse.scripts_entry import run_exp1

if __name__ == "__main__":
    raise SystemExit(run_exp1(sys.argv[1:]))
