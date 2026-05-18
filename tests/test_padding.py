"""Tests for ContextPadder using a deterministic fake tokenizer.

We avoid pulling in a real HuggingFace tokenizer here so the test suite can
run on a clean machine without network access.
"""

from __future__ import annotations

from typing import Iterable

import pytest

from long_context_discourse.padding import ContextPadder


class _WordTokenizer:
    """Whitespace-token tokenizer just for tests.

    Only implements the few methods that ``ContextPadder`` calls.
    """

    pad_token: str = "<pad>"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        return [hash(tok) & 0xFFFF for tok in text.split()] if text else []

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:  # noqa: ARG002
        # We don't have a true vocabulary; use a placeholder repeated.
        return " ".join(["w"] * len(list(token_ids)))


def test_padder_returns_target_only_when_budget_zero() -> None:
    padder = ContextPadder(_WordTokenizer(), padding_pool=["filler one", "filler two"], seed=0)
    out = padder.build("a b c", "d e f", target_token_length=2)
    assert "Argument 1: a b c" in out.context
    assert "Argument 2: d e f" in out.context
    assert "--- Focus on the following arguments ---" not in out.context  # no padding fitted


def test_padder_includes_delimiter_when_padding_added() -> None:
    padder = ContextPadder(_WordTokenizer(), padding_pool=["filler " * 50], seed=0)
    out = padder.build("hello", "world", target_token_length=64)
    assert "--- Focus on the following arguments ---" in out.context
    assert out.actual_length > 0


def test_padder_rejects_empty_pool() -> None:
    with pytest.raises(ValueError):
        ContextPadder(_WordTokenizer(), padding_pool=[], seed=0)
