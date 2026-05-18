"""Tests for the metric helpers."""

from __future__ import annotations

import pandas as pd

from long_context_discourse.metrics import (
    compute_macro_f1,
    degradation_table,
    fpar_summary,
    parse_failure_rate,
    spearman_per_model,
    wilcoxon_implicit_vs_explicit,
)


def _exp1_summary() -> pd.DataFrame:
    """Synthetic summary with implicit dropping faster than explicit on all models."""
    rows = []
    for model in ("gpt4o", "llama3_70b", "mistral_7b", "phi4", "deepseek_v3"):
        rows.append({"model": model, "rel_type": "Implicit", "context_length": 512, "macro_f1": 0.80})
        rows.append({"model": model, "rel_type": "Implicit", "context_length": 65536, "macro_f1": 0.40})
        rows.append({"model": model, "rel_type": "Explicit", "context_length": 512, "macro_f1": 0.85})
        rows.append({"model": model, "rel_type": "Explicit", "context_length": 65536, "macro_f1": 0.78})
    return pd.DataFrame(rows)


def test_compute_macro_f1_perfect() -> None:
    df = pd.DataFrame(
        {"gold_label": ["A", "B", "C", "D"], "pred_label": ["A", "B", "C", "D"]}
    )
    out = compute_macro_f1(df)
    assert out["macro_f1"] == 1.0
    assert out["accuracy"] == 1.0
    assert out["n"] == 4


def test_parse_failure_rate() -> None:
    df = pd.DataFrame({"pred_label": ["A", "X", "X", "B"]})
    assert parse_failure_rate(df) == 0.5


def test_degradation_table_uses_max_available() -> None:
    summary = _exp1_summary()
    deg = degradation_table(summary)
    impl = deg[deg["rel_type"] == "Implicit"]
    expl = deg[deg["rel_type"] == "Explicit"]
    assert len(impl) == 5 and len(expl) == 5
    # Implicit drop = (0.80 - 0.40) / 0.80 * 100 = 50.0
    assert impl["pct_drop"].iloc[0] == 50.0
    # Explicit drop ≈ (0.85 - 0.78) / 0.85 * 100 ≈ 8.24
    assert abs(expl["pct_drop"].iloc[0] - ((0.85 - 0.78) / 0.85 * 100)) < 0.01


def test_wilcoxon_returns_significant_when_implicit_higher() -> None:
    summary = _exp1_summary()
    deg = degradation_table(summary)
    stat, p = wilcoxon_implicit_vs_explicit(deg)
    assert stat is not None and p is not None
    assert p < 0.05


def test_wilcoxon_returns_none_when_too_few_models() -> None:
    rows = []
    for model in ("m1", "m2"):
        rows.append({"model": model, "rel_type": "Implicit", "delta_f1": 0.5})
        rows.append({"model": model, "rel_type": "Explicit", "delta_f1": 0.1})
    stat, p = wilcoxon_implicit_vs_explicit(pd.DataFrame(rows))
    assert stat is None and p is None


def test_fpar_and_spearman() -> None:
    rows = []
    for distance, frac_accept in zip([1, 5, 10, 20, 50], [0.0, 0.1, 0.3, 0.6, 0.8], strict=False):
        for i in range(20):
            rows.append({"model": "M", "distance": distance, "accepted": int(i < frac_accept * 20)})
    df = pd.DataFrame(rows)
    summary = fpar_summary(df)
    assert summary["fpar_pct"].max() > summary["fpar_pct"].min()
    sp = spearman_per_model(df)
    assert sp["rho"].iloc[0] > 0.5
    assert sp["p_value"].iloc[0] < 0.05
