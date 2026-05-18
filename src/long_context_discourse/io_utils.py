"""I/O helpers used across experiments.

Provides JSON / JSONL read+write with atomic semantics (write to ``*.tmp``
then rename) so a crash mid-write never leaves a half-finished result file.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing; return resolved Path."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    """Load JSON from disk, raising a clear error if the file is missing."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf and numpy/pandas sentinels so output is valid JSON (null, not `NaN`)."""
    if obj is None:
        return None
    # numpy scalars / pandas NA
    if hasattr(obj, "item") and callable(obj.item):
        try:
            obj = obj.item()
        except (ValueError, TypeError):
            pass
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def write_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Atomically write ``data`` as JSON to ``path``.

    Writes to ``<path>.tmp`` then ``os.replace`` — safe for in-flight
    notebooks and concurrent runs.
    """
    p = Path(path).expanduser()
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    data = _sanitize_for_json(data)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, p)
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON, skipping empty lines."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    out: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    """Append a single record as one JSON line. Creates parent dirs."""
    p = Path(path).expanduser()
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stream_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Iterator[dict[str, Any]]:
    """Pass each record through to disk and yield it back to the caller."""
    p = Path(path).expanduser()
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            yield record
