"""Padded-context construction for Experiments 1 and 4.

A single :class:`ContextPadder` is built once at the start of an experiment
(with a tokenizer + a long shuffled padding text), and then asked to produce
contexts of arbitrary token-budget for many ``(arg1, arg2)`` pairs.

Decisions hard-coded here, with rationale in the EMNLP draft:

* Same-domain padding is fed sequentially out of the train split — a
  shuffled super-string covers more topical diversity than naive repetition.
* The target argument pair is always placed **at the end** so context length
  is decoupled from positional bias on the probe span itself.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerBase

_DELIMITER = "\n\n--- Focus on the following arguments ---\n\n"


@dataclass(frozen=True)
class PaddedExample:
    """One ``(arg1, arg2, target_length)`` triple with its rendered context."""

    arg1: str
    arg2: str
    target_length: int
    actual_length: int
    context: str


class ContextPadder:
    """Build padded contexts of a given token budget.

    Parameters
    ----------
    tokenizer:
        Tokenizer used to count and slice token windows. The original
        experiment standardised on GPT-2 (well-documented ±~15% drift across
        production tokenizers).
    padding_pool:
        Sequence of strings (e.g. ``arg1 + " " + arg2`` from the training
        split) which are joined+shuffled to form the padding super-string.
    seed:
        Used to shuffle the padding pool so this is deterministic across
        runs.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding_pool: Sequence[str],
        *,
        seed: int = 42,
    ) -> None:
        if not padding_pool:
            raise ValueError("padding_pool must contain at least one sentence")
        self._tokenizer = tokenizer

        rng = random.Random(seed)
        ordered = list(padding_pool)
        rng.shuffle(ordered)
        joined = " ".join(s for s in ordered if s)
        self._padding_token_ids: list[int] = tokenizer.encode(joined, add_special_tokens=False)
        if not self._padding_token_ids:
            raise ValueError("padding_pool produced 0 tokens after encoding")

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_name: str,
        padding_pool: Sequence[str],
        *,
        seed: int = 42,
    ) -> "ContextPadder":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        return cls(tokenizer, padding_pool, seed=seed)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def build(self, arg1: str, arg2: str, target_token_length: int) -> PaddedExample:
        target_text = f"Argument 1: {arg1}\n\nArgument 2: {arg2}"
        target_ids = self._tokenizer.encode(target_text, add_special_tokens=False)
        delim_ids = self._tokenizer.encode(_DELIMITER, add_special_tokens=False)

        budget = target_token_length - len(target_ids) - len(delim_ids)
        if budget <= 0:
            return PaddedExample(arg1, arg2, target_token_length, len(target_ids), target_text)

        # Sample a contiguous slice of the padding pool. If budget exceeds
        # the pool size, wrap around — keeps context coherent locally even
        # when very long contexts are requested.
        n = len(self._padding_token_ids)
        if budget <= n:
            slice_ids = self._padding_token_ids[:budget]
        else:
            repeats = budget // n + 1
            slice_ids = (self._padding_token_ids * repeats)[:budget]

        padding_chunk = self._tokenizer.decode(slice_ids, skip_special_tokens=True)
        full = padding_chunk + _DELIMITER + target_text
        actual = len(self._tokenizer.encode(full, add_special_tokens=False))
        return PaddedExample(arg1, arg2, target_token_length, actual, full)
