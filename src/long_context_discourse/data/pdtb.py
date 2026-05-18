"""PDTB 3.0 loaders.

The pre-processed JSON files (``pdtb_test_balanced.json``,
``pdtb_train.json``) are produced upstream by the data-prep pipeline; this
module only validates structure and provides typed records.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..io_utils import read_json

_REQUIRED = ("rel_type", "sense_l1", "arg1", "arg2")
_VALID_TYPES = {"Implicit", "Explicit"}
_VALID_SENSES = {"Comparison", "Contingency", "Expansion", "Temporal"}


@dataclass(frozen=True)
class PdtbExample:
    """One PDTB argument pair with its discourse-relation label."""

    rel_type: str
    sense_l1: str
    arg1: str
    arg2: str
    extras: dict[str, object]

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "PdtbExample":
        for key in _REQUIRED:
            if key not in raw:
                raise ValueError(f"PDTB record missing required key {key!r}: {sorted(raw)}")
        rel_type = str(raw["rel_type"])
        sense = str(raw["sense_l1"])
        if rel_type not in _VALID_TYPES:
            raise ValueError(f"Unexpected rel_type={rel_type!r}; allowed: {_VALID_TYPES}")
        if sense not in _VALID_SENSES:
            raise ValueError(f"Unexpected sense_l1={sense!r}; allowed: {_VALID_SENSES}")
        return cls(
            rel_type=rel_type,
            sense_l1=sense,
            arg1=str(raw["arg1"]),
            arg2=str(raw["arg2"]),
            extras={k: v for k, v in raw.items() if k not in _REQUIRED},
        )


def _coerce(records: Iterable[dict[str, object]]) -> list[PdtbExample]:
    return [PdtbExample.from_dict(r) for r in records]


def load_pdtb_balanced(path: str | Path) -> list[PdtbExample]:
    """Load the 50-per-(type×sense) test set used in Experiment 1."""
    return _coerce(read_json(path))


def load_pdtb_train(path: str | Path) -> list[PdtbExample]:
    """Load the larger PDTB training split (used as a padding pool)."""
    return _coerce(read_json(path))


def padding_pool(train: Iterable[PdtbExample]) -> list[str]:
    """Concatenate ``arg1 + ' ' + arg2`` strings for use by :class:`ContextPadder`."""
    return [f"{ex.arg1} {ex.arg2}".strip() for ex in train]


def pdtb_pair_key(arg1: str, arg2: str) -> str:
    """Stable id for a PDTB argument pair (matches Exp 1 / Exp 5 subset manifests)."""
    payload = f"{arg1}\n\n{arg2}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def stratified_sample_pdtb(examples: list[PdtbExample], n_total: int, seed: int) -> list[PdtbExample]:
    """Sample ``n_total`` examples with equal counts per ``(rel_type, sense_l1)`` cell.

    The balanced test set has 8 cells (2 relation types × 4 senses). ``n_total``
    must be ≥ 8 and is split as evenly as possible across cells (remainder goes to
    the first cells in sorted key order).
    """
    if n_total < 8:
        raise ValueError(f"stratified PDTB sample needs n_total >= 8, got {n_total}")
    rng = random.Random(seed)
    by_cell: dict[tuple[str, str], list[PdtbExample]] = defaultdict(list)
    for ex in examples:
        by_cell[(ex.rel_type, ex.sense_l1)].append(ex)
    keys = sorted(by_cell.keys())
    n_cells = len(keys)
    if n_cells == 0:
        return []
    base, rem = divmod(n_total, n_cells)
    out: list[PdtbExample] = []
    for i, k in enumerate(keys):
        pool = by_cell[k][:]
        rng.shuffle(pool)
        take = base + (1 if i < rem else 0)
        take = min(take, len(pool))
        out.extend(pool[:take])
    rng.shuffle(out)
    return out


def stratified_pdtb_records_from_examples(sampled: list[PdtbExample]) -> list[dict[str, object]]:
    """Turn stratified :class:`PdtbExample` rows back into JSON-like dicts for Exp 5."""
    rows: list[dict[str, object]] = []
    for ex in sampled:
        row: dict[str, object] = {
            "rel_type": ex.rel_type,
            "sense_l1": ex.sense_l1,
            "arg1": ex.arg1,
            "arg2": ex.arg2,
        }
        row.update(ex.extras)
        rows.append(row)
    return rows
