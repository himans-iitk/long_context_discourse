"""Metric and statistical-test helpers shared by the analysis scripts."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import accuracy_score, f1_score

LABELS: tuple[str, ...] = ("A", "B", "C", "D")


def compute_macro_f1(group: pd.DataFrame) -> pd.Series:
    """Aggregate macro-F1 / accuracy / count for one ``(model, type, length)``."""
    gold = group["gold_label"].tolist()
    pred = group["pred_label"].tolist()
    f1 = f1_score(gold, pred, average="macro", labels=list(LABELS), zero_division=0)
    acc = accuracy_score(gold, pred)
    return pd.Series({"macro_f1": round(float(f1), 4), "accuracy": round(float(acc), 4), "n": len(group)})


def parse_failure_rate(df: pd.DataFrame, *, fail_token: str = "X") -> float:
    """Fraction of rows whose ``pred_label`` is the parse-failure sentinel."""
    if df.empty:
        return 0.0
    return float((df["pred_label"] == fail_token).mean())


def degradation_table(
    summary: pd.DataFrame,
    *,
    base_length: int = 512,
    candidate_max_lengths: Iterable[int] = (65536, 32768, 8192),
) -> pd.DataFrame:
    """Per-(model, rel_type) F1 drop from ``base_length`` to the longest available.

    ``candidate_max_lengths`` is searched in order; the first length present
    in the (model, rel_type) row is used as the "max" comparison point. This
    is what lets short-context models contribute Δ(512→8K) while frontier
    models contribute Δ(512→64K).
    """
    rows: list[dict[str, float | str]] = []
    for (model, rtype), block in summary.groupby(["model", "rel_type"]):
        by_len = block.set_index("context_length")["macro_f1"]
        if base_length not in by_len.index:
            continue
        f1_base = float(by_len.loc[base_length])
        f1_max = float("nan")
        chosen_len = float("nan")
        for cand in candidate_max_lengths:
            if cand in by_len.index:
                f1_max = float(by_len.loc[cand])
                chosen_len = cand
                break
        if np.isnan(f1_max):
            continue
        delta = f1_base - f1_max
        pct = (delta / f1_base * 100.0) if f1_base > 0 else 0.0
        rows.append(
            {
                "model": str(model),
                "rel_type": str(rtype),
                "max_length": chosen_len,
                "f1_base": round(f1_base, 4),
                "f1_max": round(f1_max, 4),
                "delta_f1": round(delta, 4),
                "pct_drop": round(pct, 2),
            }
        )
    return pd.DataFrame(rows)


def wilcoxon_implicit_vs_explicit(deg_df: pd.DataFrame) -> tuple[float | None, float | None]:
    """One-sided Wilcoxon: implicit Δ-F1 ≥ explicit Δ-F1 across models.

    Returns ``(statistic, p_value)`` or ``(None, None)`` if too few paired models.
    """
    if deg_df.empty or "delta_f1" not in deg_df.columns:
        return None, None
    pivot = deg_df.pivot_table(index="model", columns="rel_type", values="delta_f1", aggfunc="first")
    paired = pivot.dropna(subset=["Implicit", "Explicit"])
    if len(paired) < 3:
        return None, None
    stat, p = wilcoxon(
        paired["Implicit"].to_numpy(),
        paired["Explicit"].to_numpy(),
        alternative="greater",
    )
    return float(stat), float(p)


def fpar_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / count of ``accepted`` for each ``(model, distance)``."""
    if df.empty:
        return pd.DataFrame(columns=["model", "distance", "fpar", "fpar_std", "n", "fpar_pct"])
    out = (
        df.groupby(["model", "distance"])["accepted"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "fpar", "std": "fpar_std", "count": "n"})
    )
    out["fpar_pct"] = (out["fpar"] * 100).round(2)
    return out


def fpar_by_type(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["presup_type", "distance"])["accepted"]
        .agg(["mean", "count"])
        .reset_index()
    )


def spearman_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ between ``distance`` and ``accepted`` per model."""
    rows: list[dict[str, object]] = []
    for model, block in df.groupby("model"):
        rho_out: float | None
        p_out: float | None
        if block["distance"].nunique() < 3:
            rho_out, p_out = None, None
        else:
            rho, p = spearmanr(block["distance"], block["accepted"])
            rf = float(rho)
            pf = float(p)
            if not np.isfinite(rf) or not np.isfinite(pf):
                rho_out, p_out = None, None
            else:
                rho_out, p_out = rf, pf
        rows.append({"model": str(model), "rho": rho_out, "p_value": p_out})
    return pd.DataFrame(rows)
