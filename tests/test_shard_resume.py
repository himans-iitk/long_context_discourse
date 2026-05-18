"""Tests for shard resume helpers."""

from __future__ import annotations

from pathlib import Path

from long_context_discourse.io_utils import write_json
from long_context_discourse.shard_resume import json_row_count


def test_json_row_count(tmp_path: Path) -> None:
    fp = tmp_path / "x.json"
    write_json([{"a": 1}], fp)
    assert json_row_count(fp) == 1
    assert json_row_count(tmp_path / "missing.json") is None
