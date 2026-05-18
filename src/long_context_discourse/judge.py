"""LLM-as-judge for false-presupposition acceptance (Experiment 2)."""

from __future__ import annotations

from typing import Final

from .models import ChatMessage, OpenRouterClient
from .prompts import JUDGE_PROMPT

_REJECTED: Final[int] = 0
_ACCEPTED: Final[int] = 1
_FAILED: Final[int] = -1


def judge_presupposition_acceptance(
    client: OpenRouterClient,
    judge_model_id: str,
    *,
    model_response: str,
    truth_statement: str,
    false_presup_question: str,
    max_tokens: int = 10,
) -> int:
    """Return ``1`` if accepted, ``0`` if rejected, ``-1`` on parse failure.

    A separate function (not a method on the client) so the judge can be
    swapped for an offline rule-based scorer in unit tests.
    """
    prompt = JUDGE_PROMPT.format(
        truth_statement=truth_statement,
        false_presup_question=false_presup_question,
        model_response=model_response,
    )
    response = client.chat(
        judge_model_id,
        [ChatMessage(role="user", content=prompt)],
        max_tokens=max_tokens,
    )
    if not response.text:
        return _FAILED
    text = response.text.upper().strip()
    if "REJECTED" in text:
        return _REJECTED
    if "ACCEPTED" in text:
        return _ACCEPTED
    return _FAILED


# Re-exported sentinels so callers don't depend on the magic numbers above.
JUDGE_REJECTED: Final[int] = _REJECTED
JUDGE_ACCEPTED: Final[int] = _ACCEPTED
JUDGE_FAILED: Final[int] = _FAILED
