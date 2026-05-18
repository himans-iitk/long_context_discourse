"""Expected shard output sizes and resume checks for parallel OpenRouter runs."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

from .config import Config
from .config import load_config as load_config_file
from .data.pdtb import load_pdtb_balanced, stratified_sample_pdtb
from .data.presupposition import (
    load_presupposition_dataset,
    stratified_sample_presup_by_distance,
)
from .data.ted_mdb import load_ted_mdb
from .experiments.exp4_crosslingual import _balanced_subset
from .io_utils import read_json
from .prompts import MARKER_CONDITIONS

Exp2PairPlan = Literal["skip", "both", "exp2b_only"]


def json_row_count(path: Path) -> int | None:
    """Return list length if ``path`` is a JSON array file; else ``None``."""
    if not path.is_file():
        return None
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return None
    if isinstance(data, list):
        return len(data)
    return None


def _exp1_test_examples(cfg: Config) -> list:
    section = cfg.section("run")
    data_section = cfg.section("data")
    dataset_root = cfg.paths.dataset_root

    test_examples = load_pdtb_balanced(dataset_root / data_section["pdtb_test_path"])
    limit = section.get("limit_examples")
    stratified = bool(section.get("stratified_limit", False))
    if limit is not None:
        if stratified:
            test_examples = stratified_sample_pdtb(test_examples, int(limit), cfg.seed)
        else:
            rng = random.Random(cfg.seed)
            shuffled = list(test_examples)
            rng.shuffle(shuffled)
            test_examples = shuffled[: int(limit)]
    return test_examples


def expected_exp1_rows(cfg: Config) -> int:
    section = cfg.section("run")
    subset = section.get("models_subset") or cfg.raw.get("models_subset")
    if not subset or len(subset) != 1:
        raise ValueError("resume expects exactly one model in models_subset")
    model_name = subset[0]
    lengths = [int(x) for x in section["context_lengths"]]
    long_only = set(cfg.long_context_models)
    n_lengths = sum(1 for L in lengths if L <= 8192 or model_name in long_only)
    return len(_exp1_test_examples(cfg)) * n_lengths


def expected_exp2_rows(cfg: Config) -> int:
    section = cfg.section("run")
    data_section = cfg.section("data")
    dataset_root = cfg.paths.dataset_root

    examples = load_presupposition_dataset(dataset_root / data_section["presupposition_path"])
    limit = section.get("limit_examples")
    stratified = bool(section.get("stratified_limit", False))
    if limit is not None:
        if stratified:
            examples = stratified_sample_presup_by_distance(examples, int(limit), cfg.seed)
        else:
            rng = random.Random(cfg.seed)
            shuffled = list(examples)
            rng.shuffle(shuffled)
            examples = shuffled[: int(limit)]

    subset_models = section.get("models_subset") or cfg.raw.get("models_subset")
    if not subset_models or len(subset_models) != 1:
        raise ValueError("resume expects exactly one model in models_subset")
    return len(examples)


def expected_exp2b_rows(cfg: Config) -> int:
    section_ab = cfg.section("ablation")
    data_section = cfg.section("data")
    dataset_root = cfg.paths.dataset_root

    examples = load_presupposition_dataset(dataset_root / data_section["presupposition_path"])
    distance = int(section_ab["distance"])
    subset = [ex for ex in examples if ex.distance == distance]
    limit = section_ab.get("limit_examples")
    if limit is not None:
        rng = random.Random(cfg.seed)
        shuffled = list(subset)
        rng.shuffle(shuffled)
        subset = shuffled[: int(limit)]

    subset_models = list(section_ab["models"])
    if len(subset_models) != 1:
        raise ValueError("resume expects exactly one model in ablation.models")
    return len(subset) * len(MARKER_CONDITIONS)


def expected_exp4_rows(cfg: Config) -> int:
    section = cfg.section("run")
    data_section = cfg.section("data")
    dataset_root = cfg.paths.dataset_root

    examples = load_ted_mdb(dataset_root / data_section["ted_processed_path"])
    chosen = _balanced_subset(
        examples,
        list(section["languages"]),
        int(section.get("per_lang_per_type", 50)),
        cfg.seed,
    )
    limit = section.get("limit_examples")
    if limit is not None:
        rng = random.Random(cfg.seed)
        shuffled = list(chosen)
        rng.shuffle(shuffled)
        chosen = shuffled[: int(limit)]

    subset = cfg.raw.get("models_subset") or []
    if len(subset) != 1:
        raise ValueError("resume expects exactly one model in models_subset")
    model_name = subset[0]
    lengths = [int(x) for x in section["context_lengths"]]
    long_only = set(cfg.long_context_models)
    n_lengths = sum(1 for L in lengths if L <= 8192 or model_name in long_only)
    return len(chosen) * n_lengths


def expected_exp5_rows(cfg: Config) -> int:
    from .data.pdtb import stratified_pdtb_records_from_examples

    section = cfg.section("run")
    upstream = str(section.get("upstream_experiment", "exp1"))
    exp1_rows_path = cfg.paths.results_for(upstream) / "exp1_rows.json"
    if not exp1_rows_path.is_file():
        raise FileNotFoundError(
            f"Need merged Experiment 1 rows for Exp 5 sizing: {exp1_rows_path}"
        )

    cot_models = list(section["cot_models"])
    if len(cot_models) != 1:
        raise ValueError("resume expects exactly one model in cot_models")
    model_name = cot_models[0]

    cot_lengths = [int(x) for x in section["cot_context_lengths"]]
    long_only = set(cfg.long_context_models)
    n_lengths = sum(
        1 for L in cot_lengths if L <= 8192 or model_name in long_only
    )

    pdtb_test_path = cfg.section("data").get("pdtb_test_path") if "data" in cfg.raw else None
    if not pdtb_test_path:
        pdtb_test_path = "processed/pdtb/pdtb_test_balanced.json"
    pdtb_test_full = Path(pdtb_test_path)
    if not pdtb_test_full.is_absolute():
        pdtb_test_full = cfg.paths.dataset_root / pdtb_test_full

    limit = section.get("limit_examples")
    stratified = bool(section.get("stratified_limit", False))
    if limit is not None:
        if stratified:
            balanced = load_pdtb_balanced(pdtb_test_full)
            sampled = stratified_sample_pdtb(balanced, int(limit), cfg.seed)
            test = stratified_pdtb_records_from_examples(sampled)
        else:
            rng = random.Random(cfg.seed)
            test_list = list(read_json(pdtb_test_full))
            rng.shuffle(test_list)
            test = test_list[: int(limit)]
    else:
        test = read_json(pdtb_test_full)

    return len(test) * n_lengths


def shard_outputs_complete(cfg: Config, stage: Literal["exp1", "exp2", "exp2b", "exp4", "exp5"]) -> bool:
    out_dir = cfg.paths.results_for(cfg.experiment)
    if stage == "exp1":
        path = out_dir / "exp1_rows.json"
    elif stage == "exp2":
        path = out_dir / "exp2_rows.json"
    elif stage == "exp2b":
        path = out_dir / "exp2b_ablation.json"
    elif stage == "exp4":
        path = out_dir / "exp4_rows.json"
    elif stage == "exp5":
        path = out_dir / "exp5_cot_rows.json"
    else:
        raise ValueError(stage)

    got = json_row_count(path)
    if got is None:
        return False
    try:
        if stage == "exp1":
            expected = expected_exp1_rows(cfg)
        elif stage == "exp2":
            expected = expected_exp2_rows(cfg)
        elif stage == "exp2b":
            expected = expected_exp2b_rows(cfg)
        elif stage == "exp4":
            expected = expected_exp4_rows(cfg)
        else:
            expected = expected_exp5_rows(cfg)
    except FileNotFoundError:
        return False
    return got == expected


def plan_exp2_pair(cfg_path: Path, *, resume: bool) -> Exp2PairPlan:
    if not resume:
        return "both"
    cfg = load_config_file(cfg_path)
    try:
        e2 = shard_outputs_complete(cfg, "exp2")
        e2b = shard_outputs_complete(cfg, "exp2b")
    except ValueError:
        return "both"
    if e2 and e2b:
        return "skip"
    if e2 and not e2b:
        return "exp2b_only"
    return "both"


def skip_singleton_shard(cfg_path: Path, *, resume: bool, stage: str) -> bool:
    """Return True if subprocess should be skipped (already complete)."""
    if not resume:
        return False
    if stage not in ("exp1", "exp4", "exp5"):
        raise ValueError(f"unknown stage {stage!r}")
    cfg = load_config_file(cfg_path)
    try:
        if stage == "exp1":
            return shard_outputs_complete(cfg, "exp1")
        if stage == "exp4":
            return shard_outputs_complete(cfg, "exp4")
        return shard_outputs_complete(cfg, "exp5")
    except FileNotFoundError:
        return False
    except ValueError:
        return False
