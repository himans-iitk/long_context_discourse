"""Experiment 5 — reasoning models vs. standard, plus CoT prompt subset.

Most of the comparison is computed from Experiment 1's existing rows; the
only new API calls are the chain-of-thought subset on a handful of
standard models.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

from ..config import Config
from ..data.pdtb import (
    load_pdtb_balanced,
    load_pdtb_train,
    padding_pool,
    pdtb_pair_key,
    stratified_pdtb_records_from_examples,
    stratified_sample_pdtb,
)
from ..io_utils import ensure_dir, read_json, write_json
from ..logging_utils import get_logger
from ..models import ChatMessage, OpenRouterClient
from ..padding import ContextPadder
from ..parsing import LABEL_FAIL, extract_cot_label
from ..prompts import COT_PROMPT, SENSE_TO_LABEL

_log = get_logger(__name__)


@dataclass(frozen=True)
class Exp5Row:
    model: str
    condition: str
    rel_type: str
    sense_l1: str
    context_length: int
    gold_label: str
    pred_label: str
    correct: int


def _macro_f1(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    return float(
        f1_score(
            df["gold_label"].tolist(),
            df["pred_label"].tolist(),
            average="macro",
            labels=["A", "B", "C", "D"],
            zero_division=0,
        )
    )


def run(config: Config, *, env_path: str | Path | None = None) -> Path:
    """Run the chain-of-thought subset. Standard vs reasoning is computed in `analyze`."""
    section = config.section("run")
    cot_models = list(section["cot_models"])
    cot_lengths = [int(x) for x in section["cot_context_lengths"]]
    max_tokens = int(section.get("max_tokens_response", 300))
    limit = section.get("limit_examples")

    upstream = section.get("upstream_experiment", "exp1")
    exp1_rows_path = config.paths.results_for(upstream) / "exp1_rows.json"
    if not exp1_rows_path.is_file():
        _log.warning(
            "Upstream Exp1 rows missing (%s). Run Exp1 before analyze_exp5; "
            "continuing CoT-only run.",
            exp1_rows_path,
        )

    # Reload PDTB padded contexts deterministically: we re-derive the padded
    # context from arg1/arg2/context_length using the same padder.
    pdtb_test_path = (
        config.section("data").get("pdtb_test_path") if "data" in config.raw else None
    )
    if not pdtb_test_path:
        pdtb_test_path = "processed/pdtb/pdtb_test_balanced.json"
    pdtb_test_full = Path(pdtb_test_path)
    if not pdtb_test_full.is_absolute():
        pdtb_test_full = config.paths.dataset_root / pdtb_test_full

    stratified = bool(section.get("stratified_limit", False))
    if limit is not None:
        if stratified:
            balanced = load_pdtb_balanced(pdtb_test_full)
            sampled = stratified_sample_pdtb(balanced, int(limit), config.seed)
            test = stratified_pdtb_records_from_examples(sampled)
            _log.info(
                "limit_examples=%d stratified (8-way balance) → %d CoT test examples (seed=%d)",
                int(limit), len(test), config.seed,
            )
        else:
            rng = random.Random(config.seed)
            test_list = list(read_json(pdtb_test_full))
            rng.shuffle(test_list)
            test = test_list[: int(limit)]
            _log.info(
                "limit_examples=%d → shuffled (seed=%d) and using %d test examples for CoT",
                int(limit), config.seed, len(test),
            )
    else:
        test = read_json(pdtb_test_full)

    client = OpenRouterClient(config.openrouter, env_path=env_path)
    out_dir = ensure_dir(config.paths.results_for(config.experiment))
    subset_keys_path = out_dir / "exp5_subset_pair_keys.json"
    subset_pair_keys = [pdtb_pair_key(str(r["arg1"]), str(r["arg2"])) for r in test]
    write_json(subset_pair_keys, subset_keys_path)
    _log.info(
        "Wrote %d PDTB pair_key ids for subset-aligned analyze → %s",
        len(subset_pair_keys),
        subset_keys_path,
    )

    rows: list[Exp5Row] = []

    # We do a fresh, lightweight padded construction here — same tokenizer
    # as Exp1 — so this script is self-contained.
    train = load_pdtb_train(
        config.paths.dataset_root
        / config.raw.get("data", {}).get("pdtb_train_path", "processed/pdtb/pdtb_train.json")
    )
    padder = ContextPadder.from_pretrained(
        config.raw.get("data", {}).get("padding_tokenizer", "gpt2"),
        padding_pool(train),
        seed=config.seed,
    )

    long_only = set(config.long_context_models)
    for model_name in cot_models:
        if model_name not in config.models:
            _log.warning("Skipping unknown CoT model %s", model_name)
            continue
        model_id = config.models[model_name]
        for ctx_len in cot_lengths:
            if ctx_len > 8192 and model_name not in long_only:
                _log.info(
                    "Skipping CoT length %d for %s (not in long_context_models)",
                    ctx_len,
                    model_name,
                )
                continue
            for record in tqdm(test, desc=f"{model_name}@{ctx_len}"):
                padded = padder.build(record["arg1"], record["arg2"], ctx_len)
                response = client.chat(
                    model_id,
                    [ChatMessage(role="user", content=COT_PROMPT.format(context=padded.context))],
                    max_tokens=max_tokens,
                    temperature=config.temperature,
                )
                pred = extract_cot_label(response.text)
                gold = SENSE_TO_LABEL.get(record["sense_l1"], LABEL_FAIL)
                rows.append(
                    Exp5Row(
                        model=model_name,
                        condition="cot_prompted",
                        rel_type=record["rel_type"],
                        sense_l1=record["sense_l1"],
                        context_length=ctx_len,
                        gold_label=gold,
                        pred_label=pred,
                        correct=int(pred == gold),
                    )
                )
                client.sleep_between_calls()

    out_path = out_dir / "exp5_cot_rows.json"
    write_json([asdict(r) for r in rows], out_path)
    return out_path


def analyze(config: Config) -> dict[str, object]:
    out_dir = config.paths.results_for(config.experiment)
    cot_rows = read_json(out_dir / "exp5_cot_rows.json")
    cot_df = pd.DataFrame(cot_rows)

    upstream = config.section("run").get("upstream_experiment", "exp1")
    exp1_path = config.paths.results_for(upstream) / "exp1_rows.json"
    if exp1_path.is_file():
        exp1_rows = read_json(exp1_path)
        exp1_df = pd.DataFrame(exp1_rows)
        exp1_df = exp1_df[exp1_df["pred_label"] != LABEL_FAIL]
    else:
        _log.warning(
            "Missing upstream Exp1 rows at %s — writing exp5_summary.json with CoT macro-F1 "
            "only (standard_macro_f1 and reasoning_macro_f1 will be null). "
            "Run Exp 1 for experiment %r (merge shards if parallel), then re-run analyze_exp5.",
            exp1_path,
            upstream,
        )
        exp1_df = pd.DataFrame(
            columns=[
                "model",
                "rel_type",
                "context_length",
                "gold_label",
                "pred_label",
                "pair_key",
            ]
        )
    cot_df = cot_df[cot_df["pred_label"] != LABEL_FAIL]

    subset_path = out_dir / "exp5_subset_pair_keys.json"
    if subset_path.is_file() and not exp1_df.empty:
        keys_raw = read_json(subset_path)
        subset_keys = set(str(k) for k in keys_raw)
        pk_series = exp1_df["pair_key"].fillna("").astype(str) if "pair_key" in exp1_df.columns else None
        nonempty_pk = pk_series is not None and bool((pk_series.str.strip() != "").any())
        if nonempty_pk:
            aligned = exp1_df[exp1_df["pair_key"].isin(subset_keys)]
            if aligned.empty:
                _log.warning(
                    "Exp5 subset pair_keys did not match any Exp1 rows; using full Exp1 for comparison."
                )
            else:
                exp1_df = aligned
                _log.info(
                    "Aligned Exp1 comparison rows via pair_key (%d subset keys → %d Exp1 rows)",
                    len(subset_keys),
                    len(exp1_df),
                )
        else:
            _log.warning(
                "Found %s but upstream Exp1 rows lack nonempty pair_key — "
                "re-run Exp1 with current code (or remerge shards) for subset-aligned Exp5 summaries.",
                subset_path.name,
            )
    elif subset_path.is_file():
        _log.warning(
            "%s is present but Exp1 rows are missing — skipping pair_key alignment.",
            subset_path.name,
        )

    reasoning = set(config.reasoning_models)
    cot_models = set(config.section("run")["cot_models"])
    cot_lengths = [int(x) for x in config.section("run")["cot_context_lengths"]]

    rows: list[dict[str, object]] = []
    for ctx in cot_lengths:
        for rtype in ("Implicit", "Explicit"):
            std = exp1_df[
                (exp1_df["model"].isin(cot_models))
                & (exp1_df["rel_type"] == rtype)
                & (exp1_df["context_length"] == ctx)
            ]
            cot = cot_df[
                (cot_df["rel_type"] == rtype) & (cot_df["context_length"] == ctx)
            ]
            reason = exp1_df[
                (exp1_df["model"].isin(reasoning))
                & (exp1_df["rel_type"] == rtype)
                & (exp1_df["context_length"] == ctx)
            ]
            rows.append(
                {
                    "context_length": ctx,
                    "rel_type": rtype,
                    "standard_macro_f1": _macro_f1(std),
                    "cot_macro_f1": _macro_f1(cot),
                    "reasoning_macro_f1": _macro_f1(reason),
                }
            )
    write_json(rows, out_dir / "exp5_summary.json")
    return {"experiment": "exp5", "comparison": rows}
