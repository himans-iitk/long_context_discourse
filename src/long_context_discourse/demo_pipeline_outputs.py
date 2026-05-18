"""Synthetic **demo** artefacts matching real pipeline filenames under ``results/exp{1,2,4,5}/``.

Numbers are **not** from API runs — only for layout testing, docs, and offline plots.
Regenerate via ``scripts/bootstrap_demo_pipeline_outputs.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import ensure_dir, write_json
from .metrics import degradation_table, wilcoxon_implicit_vs_explicit

MODEL_REGISTRY: dict[str, str] = {
    "gpt4o": "openai/gpt-4o",
    "deepseek_r1": "deepseek/deepseek-r1",
    "llama3_70b": "meta-llama/llama-3.3-70b-instruct",
    "llama3_8b": "meta-llama/llama-3.1-8b-instruct",
    "mistral_7b": "mistralai/mistral-7b-instruct-v0.1",
    "mixtral": "mistralai/mixtral-8x22b-instruct",
    "phi4": "microsoft/phi-4",
    "deepseek_v3": "deepseek/deepseek-chat-v3.1",
}

ALL_MODELS: tuple[str, ...] = tuple(MODEL_REGISTRY.keys())
LONG_CTX_MODELS = frozenset({"gpt4o", "deepseek_r1", "llama3_70b"})
EXP1_LENGTHS = [512, 2048, 8192, 32768, 65536]
LABELS = ["A", "B", "C", "D"]
RNG = np.random.default_rng(42)

# Macro-F1 anchors @512 / @8192 (representative magnitudes for demo projection).
ANCHORS_512_8192: dict[str, tuple[float, float]] = {
    "gpt4o": (0.44, 0.32),
    "deepseek_r1": (0.38, 0.36),
    "llama3_70b": (0.31, 0.36),
    "llama3_8b": (0.17, 0.27),
    "mistral_7b": (0.11, 0.10),
    "mixtral": (0.21, 0.31),
    "phi4": (0.40, 0.35),
    "deepseek_v3": (0.37, 0.37),
}


def _extrapolate_f1(f512: float, f8192: float, length: int) -> float:
    if length <= 512:
        return float(np.clip(f512, 0.05, 0.55))
    if length <= 8192:
        t = (math.log2(length) - math.log2(512)) / (math.log2(8192) - math.log2(512))
        return float(np.clip(f512 + t * (f8192 - f512), 0.05, 0.55))
    extra_log = math.log2(length) - math.log2(8192)
    decay = 0.045 * extra_log + 0.012 * max(0, extra_log - 2)
    return float(np.clip(f8192 - decay - 0.01 * RNG.random(), 0.06, 0.52))


def _pair_key(i: int) -> str:
    h = hashlib.sha256(f"demo#{i}".encode()).hexdigest()
    return h[:24]


def build_exp1_summary_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for m in ALL_MODELS:
        f512, f8192 = ANCHORS_512_8192[m]
        for rt in ("Implicit", "Explicit"):
            bias = 0.03 if rt == "Explicit" else -0.03
            for L in EXP1_LENGTHS:
                if L > 8192 and m not in LONG_CTX_MODELS:
                    continue
                macro = _extrapolate_f1(f512 + bias, f8192 + bias, L)
                rows.append(
                    {
                        "model": m,
                        "rel_type": rt,
                        "context_length": L,
                        "macro_f1": round(macro, 4),
                        "accuracy": round(min(0.92, macro + 0.08), 4),
                        "n": 400,
                    }
                )
    return pd.DataFrame(rows)


def synthesize_exp1_rows(summary: pd.DataFrame, *, rows_per_cell: int = 16) -> list[dict[str, Any]]:
    """Cheap synthetic rows so ``analyze_exp1`` could be re-run coarsely."""
    out: list[dict[str, Any]] = []
    idx = 0
    for _, r in summary.iterrows():
        macro_target = float(r["macro_f1"])
        acc_prob = float(np.clip(macro_target + 0.06, 0.15, 0.85))
        senses = ["Comparison", "Contingency", "Expansion", "Temporal"]
        for _ in range(rows_per_cell):
            sense = str(RNG.choice(senses))
            gold = {"Comparison": "A", "Contingency": "B", "Expansion": "C", "Temporal": "D"}[
                sense
            ]
            pred = gold if RNG.random() < acc_prob else str(RNG.choice(LABELS))
            idx += 1
            out.append(
                {
                    "model": str(r["model"]),
                    "model_id": MODEL_REGISTRY[str(r["model"])],
                    "rel_type": str(r["rel_type"]),
                    "sense_l1": sense,
                    "context_length": int(r["context_length"]),
                    "actual_length": int(r["context_length"]),
                    "gold_label": gold,
                    "pred_label": pred,
                    "correct": int(pred == gold),
                    "raw_output": pred,
                    "arg1_words": 12,
                    "arg2_words": 14,
                    "pair_key": _pair_key(idx),
                }
            )
    return out


def write_exp1_bundle(results_dir: Path, figures_dir: Path) -> None:
    out = ensure_dir(results_dir / "exp1")
    summary = build_exp1_summary_table()
    write_json(summary.to_dict(orient="records"), out / "exp1_summary_f1.json")
    deg = degradation_table(summary)
    write_json(deg.to_dict(orient="records"), out / "exp1_degradation.json")
    stat, p_value = wilcoxon_implicit_vs_explicit(deg)
    rows_json = synthesize_exp1_rows(summary)
    write_json(rows_json, out / "exp1_rows.json")
    write_json(
        {
            "experiment": "exp1",
            "n_rows": len(rows_json),
            "parse_failure_rate": 0.015,
            "implicit_avg_degradation_pct": float(
                deg.loc[deg["rel_type"] == "Implicit", "pct_drop"].mean()
            )
            if not deg.empty
            else None,
            "explicit_avg_degradation_pct": float(
                deg.loc[deg["rel_type"] == "Explicit", "pct_drop"].mean()
            )
            if not deg.empty
            else None,
            "degradation_ratio": None,
            "wilcoxon_statistic": stat,
            "wilcoxon_p_value": p_value,
            "figure": str(figures_dir / "fig1_degradation_demo.png"),
            "_demo_synthetic": True,
        },
        out / "exp1_summary.json",
    )

    matplotlib.rcParams.update({"font.size": 10})
    key_models = ["gpt4o", "llama3_70b", "mistral_7b", "deepseek_r1"]
    ctx_vals = [512, 2048, 8192, 32768, 65536]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    axes_flat = axes.flatten()
    for ax, model_name in zip(axes_flat, key_models, strict=False):
        for rtype, color, ls in (
            ("Implicit", "#d62728", "-"),
            ("Explicit", "#1f77b4", "--"),
        ):
            ys: list[float] = []
            xs_labels: list[str] = []
            for ctx in ctx_vals:
                row = summary[
                    (summary["model"] == model_name)
                    & (summary["rel_type"] == rtype)
                    & (summary["context_length"] == ctx)
                ]
                if row.empty:
                    continue
                ys.append(float(row["macro_f1"].iloc[0]))
                xs_labels.append(str(ctx))
            if ys:
                ax.plot(xs_labels, ys, color=color, linestyle=ls, marker="o", label=rtype)
        ax.set_title(model_name)
        ax.set_ylim(0, 0.55)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Figure 1 (demo synthetic): PDTB macro-F1 vs length")
    fig.tight_layout()
    ensure_dir(figures_dir)
    fig.savefig(figures_dir / "fig1_degradation_demo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_exp2_bundle(results_dir: Path, figures_dir: Path) -> None:
    out = ensure_dir(results_dir / "exp2")
    distances = [1, 5, 10, 20, 50]
    fpar_pct: dict[str, list[float]] = {}
    base = {
        "gpt4o": 3,
        "deepseek_r1": 2,
        "llama3_70b": 2,
        "llama3_8b": 1,
        "mistral_7b": 2,
        "mixtral": 8,
        "phi4": 18,
        "deepseek_v3": 1,
    }
    for m in ALL_MODELS:
        b = base[m]
        fpar_pct[m] = [float(np.clip(b + 0.35 * d + RNG.normal(0, 1.2), 0, 72)) for d in distances]

    fpar_rows: list[dict[str, Any]] = []
    by_type_rows: list[dict[str, Any]] = []
    spearman_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    types = ["existential", "factive", "temporal", "counterfactual", "social"]
    for mi, m in enumerate(ALL_MODELS):
        rhos: list[float] = []
        for di, d in enumerate(distances):
            pct = fpar_pct[m][di]
            fpar_rows.append(
                {
                    "model": m,
                    "distance": d,
                    "fpar": pct / 100.0,
                    "fpar_std": 0.08,
                    "n": 500 // len(distances),
                    "fpar_pct": round(pct, 2),
                }
            )
            for pt in types:
                jitter = 0.85 + 0.08 * types.index(pt)
                by_type_rows.append(
                    {
                        "model": m,
                        "distance": d,
                        "presup_type": pt,
                        "mean": round((pct / 100.0) * jitter, 4),
                        "std": 0.05,
                        "n": 20,
                    }
                )
            if di > 0:
                rhos.append(pct)
        rho = float(np.corrcoef(distances, fpar_pct[m])[0, 1]) if len(distances) > 1 else 0.0
        spearman_rows.append({"model": m, "rho": round(rho, 4), "p_value": 0.05})

        for _ in range(40):
            raw_rows.append(
                {
                    "model": m,
                    "distance": int(RNG.choice(distances)),
                    "accepted": bool(RNG.random() < 0.15),
                    "presup_type": str(RNG.choice(types)),
                    "example_id": f"demo_{mi}_{len(raw_rows)}",
                }
            )

    write_json(fpar_rows, out / "exp2_fpar_summary.json")
    write_json(by_type_rows, out / "exp2_fpar_by_type.json")
    write_json(spearman_rows, out / "exp2_spearman.json")
    write_json(raw_rows, out / "exp2_rows.json")

    avg_d1 = float(np.mean([fpar_pct[m][0] for m in ALL_MODELS]))
    avg_d50 = float(np.mean([fpar_pct[m][-1] for m in ALL_MODELS]))
    write_json(
        {
            "experiment": "exp2",
            "n_rows": len(raw_rows),
            "avg_fpar_distance_1": avg_d1,
            "avg_fpar_distance_50": avg_d50,
            "rise_pp_d1_to_d50": avg_d50 - avg_d1,
            "most_vulnerable_type": "factive",
            "least_vulnerable_type": "temporal",
            "figure": str(figures_dir / "fig2_presupposition_demo.png"),
            "_demo_synthetic": True,
        },
        out / "exp2_summary.json",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for m in ALL_MODELS:
        ax.plot(distances, fpar_pct[m], marker="o", label=m)
    ax.set_xlabel("Distance")
    ax.set_ylabel("FPAR % (demo)")
    ax.legend(ncol=2, fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_presupposition_demo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    rescue_payload = {
        "experiment": "exp2b",
        "n_rows": 3200,
        "rescue": {
            "weak_marker": {"delta_pp_mean": -4.2, "delta_pp_per_model": {m: -3.5 for m in ALL_MODELS}},
            "strong_marker": {"delta_pp_mean": -9.1, "delta_pp_per_model": {m: -8.0 for m in ALL_MODELS}},
        },
        "_demo_synthetic": True,
    }
    write_json(rescue_payload, out / "exp2b_rescue.json")


def write_exp4_bundle(results_dir: Path, figures_dir: Path) -> None:
    out = ensure_dir(results_dir / "exp4")
    langs = ["English", "German", "Polish", "Russian", "Portuguese", "Turkish"]
    lengths = [512, 8192]
    summary_rows: list[dict[str, Any]] = []
    for m in ALL_MODELS:
        for lang in langs:
            for rt in ("Implicit", "Explicit"):
                for L in lengths:
                    base = 0.28 if rt == "Implicit" else 0.16
                    mu = base + (0.04 if m in {"gpt4o", "deepseek_v3"} else 0.0)
                    mu += RNG.normal(0, 0.02)
                    summary_rows.append(
                        {
                            "model": m,
                            "language": lang,
                            "rel_type": rt,
                            "context_length": L,
                            "macro_f1": round(float(np.clip(mu, 0.06, 0.55)), 4),
                            "accuracy": round(float(np.clip(mu + 0.1, 0.1, 0.75)), 4),
                            "n": 50,
                        }
                    )
    df = pd.DataFrame(summary_rows)
    write_json(summary_rows, out / "exp4_summary_f1.json")

    drops: dict[str, dict[str, dict[str, float]]] = {}
    for (lang, model_name), block in df.groupby(["language", "model"]):
        for rtype, sub in block.groupby("rel_type"):
            by_len = sub.set_index("context_length")["macro_f1"]
            if 512 not in by_len.index or 8192 not in by_len.index:
                continue
            base = float(by_len.loc[512])
            top = float(by_len.loc[8192])
            pct = (base - top) / base * 100 if base > 0 else 0.0
            drops.setdefault(str(lang), {}).setdefault(str(model_name), {})[
                str(rtype)
            ] = round(pct, 2)
    write_json(drops, out / "exp4_degradation_per_language.json")

    stub_rows = [
        {
            "model": m,
            "language": "English",
            "rel_type": "Implicit",
            "context_length": 512,
            "gold_label": "A",
            "pred_label": "A",
            "correct": 1,
        }
        for m in ALL_MODELS
    ]
    write_json(stub_rows, out / "exp4_rows.json")

    fig, ax = plt.subplots(figsize=(9, 5))
    g = df.groupby(["model", "rel_type"])["macro_f1"].mean().reset_index()
    imp = g[g["rel_type"] == "Implicit"].set_index("model").reindex(ALL_MODELS)["macro_f1"]
    ax.bar(np.arange(len(ALL_MODELS)) - 0.2, imp, width=0.4, label="Implicit")
    exp = g[g["rel_type"] == "Explicit"].set_index("model").reindex(ALL_MODELS)["macro_f1"]
    ax.bar(np.arange(len(ALL_MODELS)) + 0.2, exp, width=0.4, label="Explicit")
    ax.set_xticks(range(len(ALL_MODELS)))
    ax.set_xticklabels(list(ALL_MODELS), rotation=35, ha="right")
    ax.set_ylabel("Mean macro-F1 (demo)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig4_crosslingual_demo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_exp5_bundle(results_dir: Path, figures_dir: Path) -> None:
    out = ensure_dir(results_dir / "exp5")
    exp1 = build_exp1_summary_table()
    pooled = (
        exp1.groupby(["model", "context_length"])["macro_f1"]
        .mean()
        .reset_index()
        .rename(columns={"macro_f1": "std_f1"})
    )
    delta_map = {
        (512, "Implicit"): 0.01,
        (512, "Explicit"): 0.095,
        (8192, "Implicit"): -0.03,
        (8192, "Explicit"): 0.084,
    }
    summary_rows: list[dict[str, Any]] = []
    for L in (512, 8192):
        base_mean = float(pooled[pooled["context_length"] == L]["std_f1"].mean())
        for rt in ("Implicit", "Explicit"):
            dlt = delta_map[(L, rt)]
            cot_mean = float(np.clip(base_mean + dlt, 0.05, 0.55))
            summary_rows.append(
                {
                    "context_length": L,
                    "rel_type": rt,
                    "standard_macro_f1": round(base_mean, 4),
                    "cot_macro_f1": round(cot_mean, 4),
                    "reasoning_macro_f1": round(base_mean * 1.025, 4),
                }
            )

    cot_micro: list[dict[str, Any]] = []
    keys: list[str] = []
    z = 0
    base_mean_512 = float(pooled[pooled["context_length"] == 512]["std_f1"].mean())
    base_mean_8192 = float(pooled[pooled["context_length"] == 8192]["std_f1"].mean())
    for _, pr in pooled.iterrows():
        m = str(pr["model"])
        L = int(pr["context_length"])
        if L not in (512, 8192):
            continue
        base = float(pr["std_f1"])
        bm = base_mean_512 if L == 512 else base_mean_8192
        for rt in ("Implicit", "Explicit"):
            scale = float(np.clip(base / bm if bm > 1e-6 else 1.0, 0.85, 1.15))
            cot = float(
                np.clip(base + delta_map[(L, rt)] * scale, 0.05, 0.55)
            )
            for _ in range(8):
                z += 1
                gold = str(RNG.choice(LABELS))
                pred = gold if RNG.random() < cot else str(RNG.choice(LABELS))
                cot_micro.append(
                    {
                        "model": m,
                        "condition": "cot_prompted",
                        "rel_type": rt,
                        "sense_l1": "Comparison",
                        "context_length": L,
                        "gold_label": gold,
                        "pred_label": pred,
                        "correct": int(pred == gold),
                    }
                )
                keys.append(_pair_key(z))

    write_json(summary_rows, out / "exp5_summary.json")
    write_json(cot_micro, out / "exp5_cot_rows.json")
    write_json(keys, out / "exp5_subset_pair_keys.json")

    fig, ax = plt.subplots(figsize=(7, 4))
    sub_ex = next(
        r
        for r in summary_rows
        if r["rel_type"] == "Explicit" and r["context_length"] == 8192
    )
    ax.bar(
        [0, 1],
        [sub_ex["standard_macro_f1"], sub_ex["cot_macro_f1"]],
        color=["#555555", "#2ca02c"],
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Standard (pooled)", "CoT (pooled)"])
    ax.set_ylabel("Macro-F1 (demo)")
    ax.set_title("Exp 5 demo — Explicit @ 8192")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig5_cot_demo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_master(results_dir: Path) -> None:
    """Aggregate summaries like ``compile_master_results`` for demo."""
    payload = {
        "experiment_1": json.loads((results_dir / "exp1" / "exp1_summary.json").read_text()),
        "experiment_2": json.loads((results_dir / "exp2" / "exp2_summary.json").read_text()),
        "experiment_2b": json.loads((results_dir / "exp2" / "exp2b_rescue.json").read_text()),
        "experiment_3": None,
        "experiment_4": json.loads((results_dir / "exp4" / "exp4_summary_f1.json").read_text()),
        "experiment_5": json.loads((results_dir / "exp5" / "exp5_summary.json").read_text()),
        "metadata": {
            "models": MODEL_REGISTRY,
            "demo_synthetic": True,
            "note": "Bundled demo only — run paid configs for real EMNLP-scale numbers.",
        },
    }
    write_json(payload, results_dir / "MASTER_DEMO.json")


def emit_demo_project(project_root: Path) -> None:
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"
    write_exp1_bundle(results_dir, figures_dir)
    write_exp2_bundle(results_dir, figures_dir)
    write_exp4_bundle(results_dir, figures_dir)
    write_exp5_bundle(results_dir, figures_dir)
    write_master(results_dir)
    (results_dir / "README.txt").write_text(
        "Synthetic demo outputs (JSON + MASTER_DEMO.json). "
        "Regenerate: python scripts/bootstrap_demo_pipeline_outputs.py\n"
        "Not from OpenRouter — replace with real runs for publication.\n",
        encoding="utf-8",
    )
