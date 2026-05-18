"""Experiment 2 — presupposition tracking across conversational distance.

For each ``(model, presup_example)`` we build a multi-turn conversation,
ask the model the false-presupposition question, then have an LLM judge
classify whether the model accepted or rejected the presupposition.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import Config
from ..conversation import build_conversation
from ..data.presupposition import (
    PresupExample,
    load_presupposition_dataset,
    stratified_sample_presup_by_distance,
)
from ..io_utils import ensure_dir, read_json, write_json
from ..judge import JUDGE_FAILED, judge_presupposition_acceptance
from ..logging_utils import get_logger
from ..metrics import fpar_by_type, fpar_summary, spearman_per_model
from ..models import OpenRouterClient

_log = get_logger(__name__)


@dataclass(frozen=True)
class Exp2Row:
    model: str
    model_id: str
    example_id: str
    presup_type: str
    distance: int
    truth: str
    question: str
    response: str
    accepted: int


def run(config: Config, *, env_path: str | Path | None = None) -> Path:
    section = config.section("run")
    data_section = config.section("data")
    dataset_root = config.paths.dataset_root

    examples: list[PresupExample] = load_presupposition_dataset(
        dataset_root / data_section["presupposition_path"]
    )
    _log.info("Loaded %d presupposition examples", len(examples))

    limit = section.get("limit_examples")
    stratified = bool(section.get("stratified_limit", False))
    if limit is not None:
        if stratified:
            examples = stratified_sample_presup_by_distance(examples, int(limit), config.seed)
            _log.info(
                "limit_examples=%d stratified by distance → %d examples (seed=%d)",
                int(limit), len(examples), config.seed,
            )
        else:
            rng = random.Random(config.seed)
            shuffled = list(examples)
            rng.shuffle(shuffled)
            examples = shuffled[: int(limit)]
            _log.info(
                "limit_examples=%d → shuffled (seed=%d) and using %d examples",
                int(limit), config.seed, len(examples),
            )

    judge_model = str(section["judge_model"])
    response_tokens = int(section.get("max_tokens_response", 200))
    judge_tokens = int(section.get("max_tokens_judge", 10))
    checkpoint_every = int(section.get("checkpoint_every", 100))

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

    rows: list[Exp2Row] = []
    api_calls = 0
    for model_name, model_id in models_iter:
        _log.info("Running model %s", model_name)
        for ex in tqdm(examples, desc=model_name):
            conversation = build_conversation(
                ex.truth_statement, ex.filler_turns, ex.false_presup_question
            )
            response = client.chat(
                model_id,
                conversation,
                max_tokens=response_tokens,
                temperature=config.temperature,
            )
            if not response.text:
                continue
            verdict = judge_presupposition_acceptance(
                client,
                judge_model,
                model_response=response.text,
                truth_statement=ex.truth_statement,
                false_presup_question=ex.false_presup_question,
                max_tokens=judge_tokens,
            )
            rows.append(
                Exp2Row(
                    model=model_name,
                    model_id=model_id,
                    example_id=ex.id,
                    presup_type=ex.presup_type,
                    distance=ex.distance,
                    truth=ex.truth_statement,
                    question=ex.false_presup_question,
                    response=response.text[:512],
                    accepted=verdict,
                )
            )
            api_calls += 1
            if api_calls % checkpoint_every == 0:
                write_json([asdict(r) for r in rows], ckpt_dir / f"checkpoint_{api_calls}.json")
            client.sleep_between_calls()

        write_json(
            [asdict(r) for r in rows if r.model == model_name],
            out_dir / f"exp2_{model_name}.json",
        )

    final_path = out_dir / "exp2_rows.json"
    write_json([asdict(r) for r in rows], final_path)
    _log.info("Experiment 2 complete: %d rows → %s", len(rows), final_path)
    return final_path


def analyze(config: Config) -> dict[str, object]:
    out_dir = config.paths.results_for(config.experiment)
    rows = read_json(out_dir / "exp2_rows.json")
    df = pd.DataFrame(rows)
    df = df[df["accepted"] != JUDGE_FAILED].copy()

    fpar = fpar_summary(df)
    by_type = fpar_by_type(df)
    spearman = spearman_per_model(df)

    write_json(fpar.to_dict(orient="records"), out_dir / "exp2_fpar_summary.json")
    write_json(by_type.to_dict(orient="records"), out_dir / "exp2_fpar_by_type.json")
    write_json(spearman.to_dict(orient="records"), out_dir / "exp2_spearman.json")

    fig_path = _plot(fpar, by_type, config)

    avg_d1 = float(fpar.loc[fpar["distance"] == 1, "fpar_pct"].mean())
    avg_d50 = float(fpar.loc[fpar["distance"] == 50, "fpar_pct"].mean())
    most = (
        by_type.groupby("presup_type")["mean"].mean().idxmax()
        if not by_type.empty
        else None
    )
    least = (
        by_type.groupby("presup_type")["mean"].mean().idxmin()
        if not by_type.empty
        else None
    )

    payload: dict[str, object] = {
        "experiment": "exp2",
        "n_rows": int(len(df)),
        "avg_fpar_distance_1": avg_d1,
        "avg_fpar_distance_50": avg_d50,
        "rise_pp_d1_to_d50": (
            (avg_d50 - avg_d1) if avg_d1 is not None and avg_d50 is not None else None
        ),
        "most_vulnerable_type": str(most) if most is not None else None,
        "least_vulnerable_type": str(least) if least is not None else None,
        "figure": str(fig_path),
    }
    write_json(payload, out_dir / "exp2_summary.json")
    _log.info(
        "FPAR @1=%s  @50=%s  rise=%s",
        f"{avg_d1:.2f}%" if avg_d1 is not None else "n/a",
        f"{avg_d50:.2f}%" if avg_d50 is not None else "n/a",
        f"{avg_d50 - avg_d1:.2f}pp"
        if avg_d1 is not None and avg_d50 is not None
        else "n/a",
    )
    return payload


def _plot(fpar: pd.DataFrame, by_type: pd.DataFrame, config: Config) -> Path:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["font.size"] = 10

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    distances = sorted(fpar["distance"].unique().tolist()) or [1, 5, 10, 20, 50]

    reasoning = set(config.reasoning_models)
    for model_name in sorted(fpar["model"].unique()):
        sub = fpar[fpar["model"] == model_name].sort_values("distance")
        ls = "--" if model_name in reasoning else "-"
        lw = 3.0 if model_name in reasoning else 1.5
        ax1.plot(
            sub["distance"],
            sub["fpar_pct"],
            linestyle=ls,
            linewidth=lw,
            marker="o",
            label=model_name,
        )
    ax1.axhline(50, color="gray", linestyle=":", alpha=0.5, label="random chance")
    ax1.set_xlabel("Conversational distance (turns)")
    ax1.set_ylabel("FPAR (%)")
    ax1.set_title("FPAR by Model and Distance", fontweight="bold")
    ax1.legend(fontsize=7, ncol=2, loc="upper left")
    ax1.set_xticks(distances)
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    type_colors = {
        "existential": "#1f77b4",
        "factive": "#d62728",
        "temporal": "#ff7f0e",
        "counterfactual": "#2ca02c",
        "social": "#9467bd",
    }
    for ptype, color in type_colors.items():
        sub = by_type[by_type["presup_type"] == ptype].sort_values("distance")
        if sub.empty:
            continue
        ax2.plot(
            sub["distance"],
            sub["mean"] * 100,
            color=color,
            marker="s",
            linewidth=2.5,
            markersize=9,
            label=ptype.title(),
        )
    ax2.set_xlabel("Conversational distance (turns)")
    ax2.set_ylabel("FPAR (%)")
    ax2.set_title("FPAR by Presupposition Type", fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_xticks(distances)
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle("Figure 2: Presupposition Tracking Failure Rate", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = config.paths.figures_dir / "fig2_presupposition.pdf"
    ensure_dir(out_path.parent)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path
