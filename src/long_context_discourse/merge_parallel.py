"""Merge row-level JSON outputs from parallel experiment shards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_json, write_json

MERGE_SPECS: dict[str, dict[str, Any]] = {
    "exp1": {
        "shard_prefix": "exp1__shard_",
        "input_name": "exp1_rows.json",
        "target_dir": "exp1",
        "output_name": "exp1_rows.json",
        "sort_keys": ("model", "rel_type", "sense_l1", "context_length"),
    },
    "exp2": {
        "shard_prefix": "exp2__shard_",
        "input_name": "exp2_rows.json",
        "target_dir": "exp2",
        "output_name": "exp2_rows.json",
        "sort_keys": ("model", "example_id", "distance"),
    },
    "exp2b": {
        "shard_prefix": "exp2__shard_",
        "input_name": "exp2b_ablation.json",
        "target_dir": "exp2",
        "output_name": "exp2b_ablation.json",
        "sort_keys": ("model", "condition", "example_id"),
    },
    "exp4": {
        "shard_prefix": "exp4__shard_",
        "input_name": "exp4_rows.json",
        "target_dir": "exp4",
        "output_name": "exp4_rows.json",
        "sort_keys": ("model", "language", "rel_type", "context_length"),
    },
    "exp5": {
        "shard_prefix": "exp5__shard_",
        "input_name": "exp5_cot_rows.json",
        "target_dir": "exp5",
        "output_name": "exp5_cot_rows.json",
        "sort_keys": ("model", "context_length", "rel_type", "sense_l1"),
    },
}


def _sort_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        parts: list[Any] = []
        for k in keys:
            v = row.get(k, "")
            if k in ("context_length", "distance"):
                try:
                    parts.append(int(v))
                except (TypeError, ValueError):
                    parts.append(-1)
            else:
                parts.append(str(v))
        return tuple(parts)

    return sorted(rows, key=key)


def merge_one(
    *,
    results_dir: Path,
    shard_prefix: str,
    input_name: str,
    target_dir: str,
    output_name: str,
    sort_keys: tuple[str, ...],
) -> tuple[Path, int]:
    shards = sorted(
        p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith(shard_prefix)
    )
    if not shards:
        raise FileNotFoundError(f"No shard directories matching {shard_prefix!r} under {results_dir}")

    merged: list[dict[str, Any]] = []
    for d in shards:
        fp = d / input_name
        if not fp.is_file():
            raise FileNotFoundError(f"Missing shard output {fp}")
        chunk = read_json(fp)
        if not isinstance(chunk, list):
            raise TypeError(f"Expected list in {fp}, got {type(chunk).__name__}")
        merged.extend(chunk)

    merged = _sort_rows(merged, sort_keys)
    out_dir = ensure_dir(results_dir / target_dir)
    out_path = out_dir / output_name
    write_json(merged, out_path)
    return out_path, len(merged)


def main_merge(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge parallel shard JSON outputs.")
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--experiment", choices=sorted(MERGE_SPECS.keys()), default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)

    root = args.project_root.resolve()
    res = root / "results" if args.results_dir is None else args.results_dir.resolve()

    targets: list[str] = []
    if args.all:
        targets.extend(MERGE_SPECS.keys())
    if args.experiment:
        targets.append(args.experiment)

    seen: set[str] = set()
    ordered = [x for x in targets if not (x in seen or seen.add(x))]
    if not ordered:
        ap.error("Pass --experiment EXP or --all")

    for name in ordered:
        spec = MERGE_SPECS[name]
        path, n = merge_one(results_dir=res, **spec)
        print(f"Merged {name} -> {path} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_merge())
