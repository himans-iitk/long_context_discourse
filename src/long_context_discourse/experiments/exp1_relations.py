"""Experiment 1 — implicit vs. explicit discourse relation degradation.

Runs a multi-model sweep over ``{Implicit, Explicit} × 4 senses × 5 lengths``
on PDTB 3.0, writes per-row predictions to disk, and (in :func:`analyze`)
computes Macro-F1, degradation table, the Wilcoxon test and Figure 1.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import Config
from ..data.pdtb import (
    PdtbExample,
    load_pdtb_balanced,
    load_pdtb_train,
    padding_pool,
    pdtb_pair_key,
    stratified_sample_pdtb,
)
from ..io_utils import ensure_dir, read_json, write_json
from ..logging_utils import get_logger
from ..metrics import (
    compute_macro_f1,
    degradation_table,
    parse_failure_rate,
    wilcoxon_implicit_vs_explicit,
)
from ..models import ChatMessage, OpenRouterClient
from ..padding import ContextPadder, PaddedExample
from ..parsing import LABEL_FAIL, extract_label
from ..prompts import DISC_REL_PROMPT, SENSE_TO_LABEL

_log = get_logger(__name__)


@dataclass(frozen=True)
class Exp1Row:
    model: str
    model_id: str
    rel_type: str
    sense_l1: str
    context_length: int
    actual_length: int
    gold_label: str
    pred_label: str
    correct: int
    raw_output: str
    arg1_words: int
    arg2_words: int
    pair_key: str = ""


def _padded_iter(
    examples: list[PdtbExample], padder: ContextPadder, lengths: list[int]
) -> Iterator[tuple[PdtbExample, PaddedExample]]:
    for ex in examples:
        for length in lengths:
            yield ex, padder.build(ex.arg1, ex.arg2, length)


def run(config: Config, *, env_path: str | Path | None = None) -> Path:
    """Run the full PDTB sweep and write per-row predictions to disk."""
    section = config.section("run")
    data_section = config.section("data")
    dataset_root = config.paths.dataset_root

    test_examples = load_pdtb_balanced(dataset_root / data_section["pdtb_test_path"])
    train_examples = load_pdtb_train(dataset_root / data_section["pdtb_train_path"])
    _log.info("Loaded %d test, %d train PDTB examples", len(test_examples), len(train_examples))

    limit = section.get("limit_examples")
    stratified = bool(section.get("stratified_limit", False))
    if limit is not None:
        if stratified:
            test_examples = stratified_sample_pdtb(test_examples, int(limit), config.seed)
            _log.info(
                "limit_examples=%d stratified (8-way balance) → %d test examples (seed=%d)",
                int(limit), len(test_examples), config.seed,
            )
        else:
            rng = random.Random(config.seed)
            shuffled = list(test_examples)
            rng.shuffle(shuffled)
            test_examples = shuffled[: int(limit)]
            _log.info(
                "limit_examples=%d → shuffled (seed=%d) and using %d test examples",
                int(limit), config.seed, len(test_examples),
            )

    padder = ContextPadder.from_pretrained(
        section.get("padding_tokenizer", "gpt2"),
        padding_pool(train_examples),
        seed=config.seed,
    )

    lengths = [int(x) for x in section["context_lengths"]]
    long_only = set(config.long_context_models)
    max_tokens = int(section.get("max_tokens_response", 10))
    checkpoint_every = int(section.get("checkpoint_every", 50))

    subset_models = section.get("models_subset") or config.raw.get("models_subset")
    if subset_models:
        models_iter = [(n, config.models[n]) for n in subset_models if n in config.models]
        skipped = [n for n in subset_models if n not in config.models]
        if skipped:
            _log.warning("Skipping unknown models in models_subset: %s", skipped)
    else:
        models_iter = list(config.models.items())
    _log.info("Running %d model(s): %s", len(models_iter), [n for n, _ in models_iter])

    client = OpenRouterClient(config.openrouter, env_path=env_path)
    out_dir = ensure_dir(config.paths.results_for(config.experiment))
    ckpt_dir = ensure_dir(config.paths.checkpoints_for(config.experiment))
    rows_path = out_dir / "exp1_rows.jsonl"
    rows_path.unlink(missing_ok=True)

    rows: list[Exp1Row] = []
    api_calls = 0
    for model_name, model_id in models_iter:
        _log.info("Running model %s (%s)", model_name, model_id)
        for ex, padded in tqdm(
            list(_padded_iter(test_examples, padder, lengths)),
            desc=model_name,
        ):
            ctx_len = padded.target_length
            if ctx_len > 8192 and model_name not in long_only:
                continue

            response = client.chat(
                model_id,
                [ChatMessage(role="user", content=DISC_REL_PROMPT.format(context=padded.context))],
                max_tokens=max_tokens,
                temperature=config.temperature,
            )
            pred = extract_label(response.text)
            gold = SENSE_TO_LABEL.get(ex.sense_l1, LABEL_FAIL)
            row = Exp1Row(
                model=model_name,
                model_id=model_id,
                rel_type=ex.rel_type,
                sense_l1=ex.sense_l1,
                context_length=ctx_len,
                actual_length=padded.actual_length,
                gold_label=gold,
                pred_label=pred,
                correct=int(pred == gold),
                raw_output=(response.text or "")[:64],
                arg1_words=len(ex.arg1.split()),
                arg2_words=len(ex.arg2.split()),
                pair_key=pdtb_pair_key(ex.arg1, ex.arg2),
            )
            rows.append(row)
            api_calls += 1

            if api_calls % checkpoint_every == 0:
                ck = ckpt_dir / f"checkpoint_{api_calls}.json"
                write_json([asdict(r) for r in rows], ck)
                _log.info("checkpoint @%d → %s", api_calls, ck)

            client.sleep_between_calls()

        per_model_path = out_dir / f"exp1_{model_name}.json"
        write_json([asdict(r) for r in rows if r.model == model_name], per_model_path)

    final_path = out_dir / "exp1_rows.json"
    write_json([asdict(r) for r in rows], final_path)
    _log.info("Experiment 1 complete: %d rows → %s", len(rows), final_path)
    return final_path


def analyze(config: Config) -> dict[str, object]:
    """Compute summary stats, save analysis artefacts, render Figure 1."""
    out_dir = config.paths.results_for(config.experiment)
    fig_dir = ensure_dir(config.paths.figures_dir)
    rows = read_json(out_dir / "exp1_rows.json")
    if not rows:
        raise RuntimeError(f"No rows found at {out_dir / 'exp1_rows.json'} — did you run exp1?")

    df = pd.DataFrame(rows)
    failure_rate = parse_failure_rate(df)
    df = df[df["pred_label"] != LABEL_FAIL].copy()

    summary = (
        df.groupby(["model", "rel_type", "context_length"])
        .apply(compute_macro_f1, include_groups=False)
        .reset_index()
    )

    deg = degradation_table(summary)
    stat, p_value = wilcoxon_implicit_vs_explicit(deg)

    write_json(summary.to_dict(orient="records"), out_dir / "exp1_summary_f1.json")
    write_json(deg.to_dict(orient="records"), out_dir / "exp1_degradation.json")

    figure_path = _plot_degradation(summary, deg, config)

    if deg.empty or "pct_drop" not in deg.columns:
        impl_mean = None
        expl_mean = None
        ratio = None
        _log.warning(
            "Degradation table is empty — need at least one context_length in {8192, 32768, 65536} "
            "alongside the 512 baseline. Smoke runs with a single length will not have degradation stats."
        )
    else:
        impl_mean = float(deg.loc[deg["rel_type"] == "Implicit", "pct_drop"].mean())
        expl_mean = float(deg.loc[deg["rel_type"] == "Explicit", "pct_drop"].mean())
        ratio = (
            impl_mean / expl_mean
            if expl_mean > 0
            else None
        )

    payload: dict[str, object] = {
        "experiment": "exp1",
        "n_rows": int(len(df)),
        "parse_failure_rate": failure_rate,
        "implicit_avg_degradation_pct": impl_mean,
        "explicit_avg_degradation_pct": expl_mean,
        "degradation_ratio": ratio,
        "wilcoxon_statistic": stat,
        "wilcoxon_p_value": p_value,
        "figure": str(figure_path),
    }
    write_json(payload, out_dir / "exp1_summary.json")
    _log.info(
        "Implicit %%drop=%s explicit=%s ratio=%s wilcoxon_p=%s",
        impl_mean if impl_mean is not None else "n/a",
        expl_mean if expl_mean is not None else "n/a",
        ratio if ratio is not None else "n/a",
        p_value if p_value is not None else "n/a",
    )
    return payload


def _plot_degradation(
    summary: pd.DataFrame, deg: pd.DataFrame, config: Config
) -> Path:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["font.size"] = 10

    analysis = config.section("analysis")
    key_models = list(analysis.get("key_models", list(config.models)[:6]))
    ctx_vals = [int(x) for x in config.section("run")["context_lengths"]]
    ctx_labels = [_human_length(v) for v in ctx_vals]

    n = len(key_models)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, model_name in zip(axes_flat, key_models, strict=False):
        for rtype, color, marker, ls in (
            ("Implicit", "#d62728", "o", "-"),
            ("Explicit", "#1f77b4", "s", "--"),
        ):
            f1_vals: list[float] = []
            for ctx in ctx_vals:
                row = summary[
                    (summary["model"] == model_name)
                    & (summary["rel_type"] == rtype)
                    & (summary["context_length"] == ctx)
                ]
                f1_vals.append(float(row["macro_f1"].iloc[0]) if not row.empty else float("nan"))
            ax.plot(
                ctx_labels,
                f1_vals,
                color=color,
                marker=marker,
                linestyle=ls,
                linewidth=2.5,
                markersize=8,
                label=rtype,
            )
        ax.set_title(model_name.replace("_", "-").upper(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Context length")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9)

    for ax in axes_flat[len(key_models):]:
        ax.set_visible(False)

    plt.suptitle(
        "Figure 1: Implicit vs. Explicit Discourse Relation Recognition by Context Length",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = config.paths.figures_dir / "fig1_degradation_curves.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


def _human_length(n: int) -> str:
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)
