"""Tests for merging parallel shard JSON outputs."""

from __future__ import annotations

from pathlib import Path

from long_context_discourse.io_utils import write_json
from long_context_discourse.merge_parallel import merge_one


def test_merge_exp1_rows(tmp_path: Path) -> None:
    res = tmp_path / "results"
    (res / "exp1__shard_a").mkdir(parents=True)
    (res / "exp1__shard_b").mkdir(parents=True)
    write_json(
        [
            {
                "model": "b",
                "rel_type": "Explicit",
                "sense_l1": "Temporal",
                "context_length": 512,
                "gold_label": "A",
                "pred_label": "A",
            }
        ],
        res / "exp1__shard_b" / "exp1_rows.json",
    )
    write_json(
        [
            {
                "model": "a",
                "rel_type": "Implicit",
                "sense_l1": "Temporal",
                "context_length": 8192,
                "gold_label": "B",
                "pred_label": "B",
            }
        ],
        res / "exp1__shard_a" / "exp1_rows.json",
    )

    out, n = merge_one(
        results_dir=res,
        shard_prefix="exp1__shard_",
        input_name="exp1_rows.json",
        target_dir="exp1",
        output_name="exp1_rows.json",
        sort_keys=("model", "rel_type", "sense_l1", "context_length"),
    )
    assert n == 2
    assert out == res / "exp1" / "exp1_rows.json"
    assert out.is_file()
