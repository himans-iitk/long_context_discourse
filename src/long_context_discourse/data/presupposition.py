"""Loader for the synthetic presupposition tracking dataset (Experiment 2)."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..io_utils import read_json

_REQUIRED = (
    "id",
    "presup_type",
    "distance",
    "truth_statement",
    "filler_turns",
    "false_presup_question",
)
_VALID_TYPES = {"existential", "factive", "temporal", "counterfactual", "social"}


@dataclass(frozen=True)
class PresupExample:
    id: str
    presup_type: str
    distance: int
    truth_statement: str
    filler_turns: str
    false_presup_question: str
    entity: str | None
    filler_topic: str | None
    extras: dict[str, object]

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "PresupExample":
        for key in _REQUIRED:
            if key not in raw:
                raise ValueError(f"Presup record missing key {key!r}: {sorted(raw)}")
        ptype = str(raw["presup_type"])
        if ptype not in _VALID_TYPES:
            raise ValueError(f"Unexpected presup_type={ptype!r}; allowed: {_VALID_TYPES}")
        return cls(
            id=str(raw["id"]),
            presup_type=ptype,
            distance=int(raw["distance"]),
            truth_statement=str(raw["truth_statement"]),
            filler_turns=str(raw["filler_turns"]),
            false_presup_question=str(raw["false_presup_question"]),
            entity=(str(raw["entity"]) if raw.get("entity") is not None else None),
            filler_topic=(str(raw["filler_topic"]) if raw.get("filler_topic") is not None else None),
            extras={
                k: v
                for k, v in raw.items()
                if k not in (*_REQUIRED, "entity", "filler_topic")
            },
        )


def load_presupposition_dataset(path: str | Path) -> list[PresupExample]:
    return [PresupExample.from_dict(r) for r in read_json(path)]


def stratified_sample_presup_by_distance(
    examples: list[PresupExample], n_total: int, seed: int
) -> list[PresupExample]:
    """Sample ``n_total`` examples with (as far as possible) equal counts per ``distance``.

    Keeps multiple distance levels represented so per-model Spearman ρ is defined more often.
    """
    by_d: dict[int, list[PresupExample]] = defaultdict(list)
    for ex in examples:
        by_d[ex.distance].append(ex)
    distances = sorted(by_d.keys())
    n_dist = len(distances)
    if n_dist == 0:
        return []
    if n_total < n_dist:
        raise ValueError(
            f"stratified presup sample needs n_total >= number of distance levels ({n_dist}), got {n_total}"
        )
    rng = random.Random(seed)
    base, rem = divmod(n_total, n_dist)
    out: list[PresupExample] = []
    for i, d in enumerate(distances):
        pool = by_d[d][:]
        rng.shuffle(pool)
        take = base + (1 if i < rem else 0)
        take = min(take, len(pool))
        out.extend(pool[:take])
    rng.shuffle(out)
    return out
