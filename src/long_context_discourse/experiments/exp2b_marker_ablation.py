"""Experiment 2B — discourse-marker rescue ablation.

Tests four conditions (no/weak/strong/repeated marker) at a single
conversational distance using a small subset of models, to measure how
much an explicit marker reduces FPAR.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import Config
from ..conversation import build_conversation_with_marker
from ..data.presupposition import load_presupposition_dataset
from ..io_utils import ensure_dir, read_json, write_json
from ..judge import JUDGE_FAILED, judge_presupposition_acceptance
from ..logging_utils import get_logger
from ..models import OpenRouterClient
from ..prompts import MARKER_CONDITIONS

_log = get_logger(__name__)


@dataclass(frozen=True)
class Exp2BRow:
    model: str
    condition: str
    presup_type: str
    distance: int
    example_id: str
    accepted: int


def run(config: Config, *, env_path: str | Path | None = None) -> Path:
    section = config.section("ablation")
    if not section.get("enabled", True):
        _log.info("Experiment 2B disabled in config")
        return Path()

    data_section = config.section("data")
    dataset_root = config.paths.dataset_root
    examples = load_presupposition_dataset(dataset_root / data_section["presupposition_path"])

    distance = int(section["distance"])
    reminder_every = int(section.get("reminder_every", 10))
    subset_models = list(section["models"])
    subset = [ex for ex in examples if ex.distance == distance]
    limit = section.get("limit_examples")
    if limit is not None:
        rng = random.Random(config.seed)
        shuffled = list(subset)
        rng.shuffle(shuffled)
        subset = shuffled[: int(limit)]
        _log.info(
            "limit_examples=%d → shuffled (seed=%d) at distance=%d",
            int(limit), config.seed, distance,
        )
    _log.info("2B subset: %d examples at distance=%d, %d models", len(subset), distance, len(subset_models))

    judge_model = str(config.section("run")["judge_model"])
    response_tokens = int(config.section("run").get("max_tokens_response", 200))
    judge_tokens = int(config.section("run").get("max_tokens_judge", 10))

    client = OpenRouterClient(config.openrouter, env_path=env_path)
    out_dir = ensure_dir(config.paths.results_for(config.experiment))

    rows: list[Exp2BRow] = []
    for model_name in subset_models:
        if model_name not in config.models:
            _log.warning("Skipping unknown model %s", model_name)
            continue
        model_id = config.models[model_name]
        for condition in MARKER_CONDITIONS:
            for ex in tqdm(subset, desc=f"{model_name}/{condition}"):
                conversation = build_conversation_with_marker(
                    ex.truth_statement,
                    ex.filler_turns,
                    ex.false_presup_question,
                    condition,
                    reminder_every=reminder_every,
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
                    Exp2BRow(
                        model=model_name,
                        condition=condition,
                        presup_type=ex.presup_type,
                        distance=distance,
                        example_id=ex.id,
                        accepted=verdict,
                    )
                )
                client.sleep_between_calls()

    out_path = out_dir / "exp2b_ablation.json"
    write_json([asdict(r) for r in rows], out_path)
    _log.info("Experiment 2B complete: %d rows → %s", len(rows), out_path)
    return out_path


def analyze(config: Config) -> dict[str, object]:
    out_dir = config.paths.results_for(config.experiment)
    path = out_dir / "exp2b_ablation.json"
    if not path.is_file():
        raise FileNotFoundError(f"No 2B output at {path}")
    df = pd.DataFrame(read_json(path))
    df = df[df["accepted"] != JUDGE_FAILED].copy()
    if df.empty:
        return {"experiment": "exp2b", "n_rows": 0}

    summary = (
        df.groupby(["model", "condition"])["accepted"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "fpar", "count": "n"})
    )
    summary["fpar_pct"] = (summary["fpar"] * 100).round(2)
    write_json(summary.to_dict(orient="records"), out_dir / "exp2b_summary.json")

    base = summary[summary["condition"] == "no_marker"].set_index("model")["fpar_pct"]
    rescue: dict[str, dict[str, float]] = {}
    for cond in summary["condition"].unique():
        if cond == "no_marker":
            continue
        m = summary[summary["condition"] == cond].set_index("model")["fpar_pct"]
        joined = base.align(m, join="inner")
        rescue[cond] = {
            "delta_pp_mean": float((joined[1] - joined[0]).mean()),
            "delta_pp_per_model": {k: float(v) for k, v in (joined[1] - joined[0]).items()},
        }
    payload = {"experiment": "exp2b", "n_rows": int(len(df)), "rescue": rescue}
    write_json(payload, out_dir / "exp2b_rescue.json")
    return payload
