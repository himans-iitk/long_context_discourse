"""Tests for JSON write sanitization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from long_context_discourse.io_utils import read_json, write_json


def test_write_json_replaces_nan(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    write_json({"a": float("nan"), "b": [1.0, float("inf")]}, path)
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    obj = json.loads(raw)
    assert obj["a"] is None
    assert obj["b"] == [1.0, None]


def test_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "y.json"
    write_json({"ok": True}, path)
    assert read_json(path) == {"ok": True}
