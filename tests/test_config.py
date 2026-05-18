"""Tests for the YAML config loader (extends + deep-merge)."""

from __future__ import annotations

from pathlib import Path

import pytest

from long_context_discourse.config import ConfigError, load_config


def _write(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_extends_deep_merges(tmp_path: Path) -> None:
    parent = tmp_path / "configs" / "default.yaml"
    parent.parent.mkdir()
    _write(
        parent,
        """
paths:
  dataset_root: data
  results_dir: results
  figures_dir: figures
openrouter:
  base_url: https://example.test
  api_key_env: TEST_KEY
  referer_env: TEST_REF
  title_env: TEST_TITLE
  request_timeout_seconds: 30
  max_retries: 1
  retry_base_seconds: 0.1
  rate_limit_sleep_seconds: 0
models:
  m1: org/m1
  m2: org/m2
""",
    )
    child = tmp_path / "configs" / "child.yaml"
    _write(
        child,
        """
extends: default.yaml
experiment: child_exp
description: smoke test
models:
  m2: override/m2
""",
    )

    import os

    os.environ["TEST_KEY"] = "k"
    config = load_config(child)
    assert config.experiment == "child_exp"
    assert config.models["m1"] == "org/m1"
    assert config.models["m2"] == "override/m2"
    assert config.openrouter.api_key_env == "TEST_KEY"


def test_missing_models_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    _write(cfg, "experiment: x\n")
    with pytest.raises(ConfigError):
        load_config(cfg)
