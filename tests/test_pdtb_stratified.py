"""Stratified sampling helpers for capped smoke runs."""

from __future__ import annotations

from collections import Counter

from long_context_discourse.data.pdtb import PdtbExample, stratified_sample_pdtb
from long_context_discourse.data.presupposition import (
    PresupExample,
    stratified_sample_presup_by_distance,
)


def test_stratified_pdtb_balances_eight_cells() -> None:
    cells: list[PdtbExample] = []
    for rt in ("Implicit", "Explicit"):
        for s in ("Comparison", "Contingency", "Expansion", "Temporal"):
            for i in range(30):
                cells.append(PdtbExample(rt, s, f"a{i}", f"b{i}", {}))
    sampled = stratified_sample_pdtb(cells, 120, seed=42)
    assert len(sampled) == 120
    counts = Counter((x.rel_type, x.sense_l1) for x in sampled)
    assert len(counts) == 8
    assert all(v == 15 for v in counts.values())


def test_stratified_presup_balances_distances() -> None:
    examples: list[PresupExample] = []
    for d in (1, 5, 10, 20, 50):
        for i in range(40):
            examples.append(
                PresupExample(
                    id=f"{d}_{i}",
                    presup_type="existential",
                    distance=d,
                    truth_statement="t",
                    filler_turns="f",
                    false_presup_question="q",
                    entity=None,
                    filler_topic=None,
                    extras={},
                )
            )
    sampled = stratified_sample_presup_by_distance(examples, 30, seed=0)
    assert len(sampled) == 30
    by_d = Counter(ex.distance for ex in sampled)
    assert set(by_d.keys()) == {1, 5, 10, 20, 50}
    assert all(v == 6 for v in by_d.values())
