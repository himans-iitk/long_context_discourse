"""Tests for label-extraction utilities."""

from __future__ import annotations

import pytest

from long_context_discourse.parsing import LABEL_FAIL, extract_cot_label, extract_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A", "A"),
        ("a", "A"),
        ("The answer is C.", "C"),
        ("**B**", "B"),
        (" \nD\n", "D"),
        ("", LABEL_FAIL),
        (None, LABEL_FAIL),
        ("nothing valid here", LABEL_FAIL),
    ],
)
def test_extract_label(raw: str | None, expected: str) -> None:
    assert extract_label(raw) == expected


def test_extract_label_picks_first_letter() -> None:
    assert extract_label("Choice: B then maybe A") == "B"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Reasoning blah\nANSWER: A", "A"),
        ("Reasoning blah\nanswer: c", "C"),
        ("...we say A initially, but final ANSWER: D", "D"),
        # No explicit ANSWER: → take last valid letter
        ("Maybe A or B but ultimately C", "C"),
        ("", LABEL_FAIL),
    ],
)
def test_extract_cot_label(raw: str, expected: str) -> None:
    assert extract_cot_label(raw) == expected
