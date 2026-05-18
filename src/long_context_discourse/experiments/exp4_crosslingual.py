"""Experiment 4 — cross-lingual discourse degradation on TED-MDB.

A lighter version of Experiment 1: 6 languages x {Implicit, Explicit} x up to 3
context lengths x a subset of models. Padding is built from the same English
PDTB pool by default (acknowledged limitation in the paper); pass a
different ``paths`` block in the config if a language-matched padding pool
becomes available.

Analysis aggregates macro-F1 **per model** (not pooled across models).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import Config
from ..data.pdtb import load_pdtb_train, padding_pool
from ..data.ted_mdb import TedMdbExample, load_ted_mdb
from ..io_utils import ensure_dir, read_json, write_json
from ..logging_utils import get_logger
from ..metrics import compute_macro_f1
from ..models import ChatMessage, OpenRouterClient
from ..padding import ContextPadder
from ..parsing import LABEL_FAIL, extract_label
from ..prompts import DISC_REL_PROMPT, SENSE_TO_LABEL

_log = get_logger(__name__)


@dataclass(frozen=True)
class Exp4Row:
    model: str
    model_id: str
    language: str
    rel_type: str
    context_length: int
    gold_label: str
    pred_label: str
    correct: int


def _balanced_subset(
    examples: list[TedMdbExample],
    languages: list[str],
    per_lang_per_type: int,
    seed: int,
) -> list[TedMdbExample]:
    rng = random.Random(seed)
    chosen: list[TedMdbExample] = []
    for lang in languages:
        in_lang = [ex for ex in examples if ex.language == lang]
        for rel_type in ("Implicit", "Explicit"):
            pool = [ex for ex in in_lang if ex.rel_type == rel_type]
            if not pool:
                continue
            n = min(per_lang_per_type, len(pool))
            chosen.extend(rng.sample(pool, n))
    return chosen


def run(config: Config, *, env_path: str | Path | None = None) -> Path:
    section = config.section("run")
    data_section = config.section("data")
    dataset_root = config.paths.dataset_root

    examples = load_ted_mdb(dataset_root / data_section["ted_processed_path"])
    train = load_pdtb_train(dataset_root / data_section["pdtb_train_path"])

    chosen = _balanced_subset(
        examples,
        list(section["languages"]),
        int(section.get("per_lang_per_type", 50)),
        config.seed,
    )
    limit = section.get("limit_examples")
    if limit is not None:
        rng = random.Random(config.seed)
        shuffled = list(chosen)
        rng.shuffle(shuffled)
        chosen = shuffled[: int(limit)]
    _log.info("Selected %d TED-MDB examples across %d languages", len(chosen), len(section["languages"]))

    padder = ContextPadder.from_pretrained(
        section.get("padding_tokenizer", "gpt2"),
        padding_pool(train),
        seed=config.seed,
    )

    lengths = [int(x) for x in section["context_lengths"]]
    long_only = set(config.long_context_models)
    max_tokens = int(section.get("max_tokens_response", 5))

    subset_models = list(config.raw.get("models_subset", config.models.keys()))
    client = OpenRouterClient(config.openrouter, env_path=env_path)
    out_dir = ensure_dir(config.paths.results_for(config.experiment))

    rows: list[Exp4Row] = []
    for model_name in subset_models:
        if model_name not in config.models:
            _log.warning("Skipping unknown model %s", model_name)
            continue
        model_id = config.models[model_name]
        for ex in tqdm(chosen, desc=model_name):
            for ctx_len in lengths:
                if ctx_len > 8192 and model_name not in long_only:
                    continue
                padded = padder.build(ex.arg1, ex.arg2, ctx_len)
                response = client.chat(
                    model_id,
                    [ChatMessage(role="user", content=DISC_REL_PROMPT.format(context=padded.context))],
                    max_tokens=max_tokens,
                    temperature=config.temperature,
                )
                pred = extract_label(response.text)
                gold = SENSE_TO_LABEL.get(ex.sense_l1, LABEL_FAIL)
                rows.append(
                    Exp4Row(
                        model=model_name,
                        model_id=model_id,
                        language=ex.language,
                        rel_type=ex.rel_type,
                        context_length=ctx_len,
                        gold_label=gold,
                        pred_label=pred,
                        correct=int(pred == gold),
                    )
                )
                client.sleep_between_calls()

    out_path = out_dir / "exp4_rows.json"
    write_json([asdict(r) for r in rows], out_path)
    _log.info("Experiment 4 complete: %d rows → %s", len(rows), out_path)
    return out_path


def analyze(config: Config) -> dict[str, object]:
    out_dir = config.paths.results_for(config.experiment)
    rows = read_json(out_dir / "exp4_rows.json")
    df = pd.DataFrame(rows)
    df = df[df["pred_label"] != LABEL_FAIL].copy()
    summary = (
        df.groupby(["model", "language", "rel_type", "context_length"])
        .apply(compute_macro_f1, include_groups=False)
        .reset_index()
    )
    write_json(summary.to_dict(orient="records"), out_dir / "exp4_summary_f1.json")

    # Per-language, per-model degradation: 512 → max available length (8192 or 32768).
    drops: dict[str, dict[str, dict[str, float]]] = {}
    for (lang, model_name), block in summary.groupby(["language", "model"]):
        for rtype, sub in block.groupby("rel_type"):
            by_len = sub.set_index("context_length")["macro_f1"]
            if 512 not in by_len.index:
                continue
            for cand in (32768, 8192):
                if cand in by_len.index:
                    base = float(by_len.loc[512])
                    top = float(by_len.loc[cand])
                    pct = (base - top) / base * 100 if base > 0 else 0.0
                    drops.setdefault(lang, {}).setdefault(str(model_name), {})[
                        str(rtype)
                    ] = round(pct, 2)
                    break
    write_json(drops, out_dir / "exp4_degradation_per_language.json")
    return {"experiment": "exp4", "n_rows": len(df), "degradation": drops}
