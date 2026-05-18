"""Robust label extraction from free-form model outputs.

We want ``"The answer is C."`` to extract ``"C"`` but ``"nothing valid here"``
to fail (the ``A`` inside ``valid`` must not match). The matching therefore
uses a word-boundary regex rather than scanning every character.
"""

from __future__ import annotations

import re
from typing import Final

LABEL_FAIL: Final[str] = "X"
_ANSWER_PATTERN: Final[re.Pattern[str]] = re.compile(r"ANSWER:\s*([ABCD])", re.IGNORECASE)
_STANDALONE_LABEL: Final[re.Pattern[str]] = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def extract_label(raw_output: str | None) -> str:
    """Return the first standalone ``A/B/C/D`` in ``raw_output``.

    A "standalone" letter is one that is not embedded in a longer word
    (so ``"answer"`` does not yield ``A``). Falls back to ``"X"`` if no
    valid letter is found, letting callers distinguish a wrong answer
    from an unparseable one.
    """
    if not raw_output:
        return LABEL_FAIL
    match = _STANDALONE_LABEL.search(raw_output.upper())
    return match.group(1) if match else LABEL_FAIL


def extract_cot_label(raw_output: str | None) -> str:
    """Extract the chain-of-thought answer.

    Prefers an explicit ``ANSWER: X`` line (case-insensitive); falls back to
    the **last** standalone ``A/B/C/D`` in the output, on the assumption
    that chain-of-thought traces often mention multiple letters while
    reasoning before committing to a final one.
    """
    if not raw_output:
        return LABEL_FAIL
    explicit = _ANSWER_PATTERN.search(raw_output)
    if explicit:
        return explicit.group(1).upper()
    matches = _STANDALONE_LABEL.findall(raw_output.upper())
    return matches[-1] if matches else LABEL_FAIL
