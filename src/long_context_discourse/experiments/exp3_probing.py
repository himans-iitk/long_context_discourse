"""Experiment 3 — mechanistic probing of nucleus/satellite representations.

Workflow:

1. Load GUM segments (text + role + position_ratio).
2. Optionally rebalance classes (config-controlled).
3. For each transformer layer, extract mean-pooled hidden states and run a
   stratified-5-fold logistic regression probe.
4. At the best layer, repeat the probe within document-position buckets.

This module does **not** import torch at module load; the heavy import is
deferred to :class:`HiddenStateExtractor`. Install ``.[probe]`` first.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm

from ..config import Config
from ..data.gum import load_gum_segments
from ..io_utils import ensure_dir, read_json, write_json
from ..logging_utils import get_logger
from ..probing import (
    HiddenStateExtractor,
    LayerProbeResult,
    PositionProbeResult,
    cross_validated_logreg,
    probe_by_position,
)

_log = get_logger(__name__)


def run(config: Config) -> Path:
    section = config.section("probe")
    data_section = config.section("data")
    dataset_root = config.paths.dataset_root

    segments = load_gum_segments(
        dataset_root / data_section["gum_processed_path"],
        min_chars=int(data_section.get("min_text_chars", 20)),
        max_chars=int(data_section.get("max_text_chars", 500)),
    )
    _log.info("Loaded %d GUM segments", len(segments))

    n_max = data_section.get("n_segments_max")
    if n_max is not None:
        segments = segments[: int(n_max)]
        _log.info("n_segments_max=%d → using %d segments", int(n_max), len(segments))

    texts = [s.text for s in segments]
    labels = np.array([s.label for s in segments])
    positions = np.array([s.position_ratio for s in segments], dtype=float)

    threshold = float(section.get("balance_threshold", 0.7))
    if (labels == 1).mean() > threshold or (labels == 0).mean() > threshold:
        nucleus_idx = np.where(labels == 1)[0]
        satellite_idx = np.where(labels == 0)[0]
        n = min(len(nucleus_idx), len(satellite_idx))
        nucleus_idx = resample(nucleus_idx, n_samples=n, random_state=config.seed)
        satellite_idx = resample(satellite_idx, n_samples=n, random_state=config.seed)
        keep = np.concatenate([nucleus_idx, satellite_idx])
        rng = np.random.default_rng(config.seed)
        rng.shuffle(keep)
        texts = [texts[i] for i in keep]
        labels = labels[keep]
        positions = positions[keep]
        _log.info("Rebalanced to %d spans (50/50)", len(texts))

    extractor = HiddenStateExtractor(
        section["model_name"],
        load_in_4bit=bool(section.get("load_in_4bit", True)),
        device_map=str(section.get("device", "auto")),
        max_length=int(section.get("max_length_tokens", 128)),
    )

    n_splits = int(section.get("cv_folds", 5))
    layers_subset = section.get("layers_subset")
    if layers_subset is not None:
        layer_indices = [int(x) for x in layers_subset]
    else:
        layer_indices = list(range(extractor.num_layers))
    _log.info("Probing %d layer(s): %s", len(layer_indices), layer_indices)

    layer_results: list[LayerProbeResult] = []
    for layer_idx in layer_indices:
        X = np.stack(
            [extractor.embed_span(t, layer_idx=layer_idx) for t in tqdm(texts, desc=f"L{layer_idx}", leave=False)]
        )
        mean, std, mx = cross_validated_logreg(X, labels, n_splits=n_splits, seed=config.seed)
        layer_results.append(
            LayerProbeResult(layer=layer_idx, mean_acc=mean, std_acc=std, max_acc=mx)
        )
        _log.info("layer=%d acc=%.3f±%.3f", layer_idx, mean, std)

    best = max(layer_results, key=lambda r: r.mean_acc)
    _log.info("Best layer: %d (acc=%.3f)", best.layer, best.mean_acc)

    best_embeddings = np.stack(
        [
            extractor.embed_span(t, layer_idx=best.layer)
            for t in tqdm(texts, desc=f"best L{best.layer}")
        ]
    )

    pos_results: list[PositionProbeResult] = probe_by_position(
        best_embeddings,
        labels,
        positions,
        n_buckets=int(section.get("position_buckets", 5)),
        n_splits=n_splits,
        seed=config.seed,
    )

    out_dir = ensure_dir(config.paths.results_for(config.experiment))
    write_json([r.as_dict() for r in layer_results], out_dir / "exp3_layer_probe.json")
    write_json([r.as_dict() for r in pos_results], out_dir / "exp3_position_probe.json")
    write_json(
        {
            "best_layer": best.layer,
            "best_mean_acc": best.mean_acc,
            "best_std_acc": best.std_acc,
            "model_name": section["model_name"],
            "n_spans": len(texts),
        },
        out_dir / "exp3_summary.json",
    )

    fig_path = _plot(layer_results, pos_results, best.layer, config)
    return fig_path


def analyze(config: Config) -> dict[str, object]:
    out_dir = config.paths.results_for(config.experiment)
    layer_rows = read_json(out_dir / "exp3_layer_probe.json")
    pos_rows = read_json(out_dir / "exp3_position_probe.json")

    best = max(layer_rows, key=lambda r: r["mean_acc"])
    starts = [r for r in pos_rows if r["bucket"] == 0]
    middles = [r for r in pos_rows if r["bucket"] == 2]
    ends = [r for r in pos_rows if r["bucket"] == max(rr["bucket"] for rr in pos_rows)]

    start_acc = starts[0]["mean_acc"] if starts else float("nan")
    middle_acc = middles[0]["mean_acc"] if middles else float("nan")
    end_acc = ends[0]["mean_acc"] if ends else float("nan")
    drop_pct = (
        (start_acc - middle_acc) / start_acc * 100 if start_acc and not np.isnan(start_acc) else float("nan")
    )

    payload: dict[str, object] = {
        "experiment": "exp3",
        "best_probing_layer": int(best["layer"]),
        "best_probing_accuracy": float(best["mean_acc"]),
        "start_position_accuracy": float(start_acc),
        "middle_position_accuracy": float(middle_acc),
        "end_position_accuracy": float(end_acc),
        "drop_start_to_middle_pct": float(drop_pct),
    }
    write_json(payload, out_dir / "exp3_analysis.json")
    return payload


def _plot(
    layer_results: list[LayerProbeResult],
    pos_results: list[PositionProbeResult],
    best_layer: int,
    config: Config,
) -> Path:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["font.size"] = 10

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    layers = [r.layer for r in layer_results]
    means = [r.mean_acc for r in layer_results]
    stds = [r.std_acc for r in layer_results]
    ax1.plot(layers, means, color="#2ca02c", linewidth=2.5, marker="o", markersize=4)
    ax1.fill_between(layers, [m - s for m, s in zip(means, stds, strict=False)], [m + s for m, s in zip(means, stds, strict=False)], alpha=0.2, color="#2ca02c")
    ax1.axvline(best_layer, color="red", linestyle="--", alpha=0.7, label=f"best L{best_layer}")
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="chance")
    ax1.set_xlabel("Transformer layer")
    ax1.set_ylabel("Probing accuracy (5-fold CV)")
    ax1.set_title("Nucleus/Satellite probe by layer", fontweight="bold")
    ax1.set_ylim(0.4, 1.0)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    if pos_results:
        labels = [r.position_label for r in pos_results]
        accs = [r.mean_acc for r in pos_results]
        stds = [r.std_acc for r in pos_results]
        colors = ["#2ca02c", "#98df8a", "#d62728", "#98df8a", "#2ca02c"][: len(accs)]
        ax2.bar(labels, accs, color=colors, yerr=stds, capsize=5, edgecolor="black", linewidth=0.5)
        ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="chance")
        ax2.set_ylim(0.4, 1.0)
        ax2.legend(fontsize=9)
    ax2.set_xlabel("Document position")
    ax2.set_ylabel("Probing accuracy")
    ax2.set_title("Probe accuracy by document position", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Figure 3: Probing of Discourse Role Representations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = ensure_dir(config.paths.figures_dir) / "fig3_probing.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path
