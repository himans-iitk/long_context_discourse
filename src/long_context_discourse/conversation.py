"""Multi-turn conversation builders for the presupposition experiment.

Both :func:`build_conversation` and :func:`build_conversation_with_marker`
produce ``list[ChatMessage]`` ready to feed into
:meth:`OpenRouterClient.chat`. Filler text is split on newlines (the format
of the synthetic dataset) and any leading speaker prefix (``A:``, ``B:``)
is stripped.
"""

from __future__ import annotations

from typing import Final

from .models import ChatMessage
from .prompts import MARKER_CONDITIONS

_ACK_TEXT: Final[str] = "Got it, I will keep that in mind."


def _strip_speaker_prefix(line: str) -> str:
    """Drop a leading ``A:`` / ``B:`` style speaker tag if present."""
    if ":" in line:
        head, rest = line.split(":", 1)
        if len(head.strip()) <= 2:
            return rest.strip()
    return line.strip()


def _split_filler(filler: str) -> list[str]:
    return [stripped for line in filler.splitlines() if (stripped := _strip_speaker_prefix(line))]


def build_conversation(
    truth_statement: str,
    filler_turns: str,
    false_presup_question: str,
    *,
    truth_prefix: str = "Just so you know: ",
) -> list[ChatMessage]:
    """Compose the full multi-turn conversation for one example.

    Structure:

    1. user: ``"<truth_prefix><truth_statement>"``
    2. assistant: acknowledgement
    3. filler turns (alternating user/assistant by line index)
    4. user: ``false_presup_question`` (replaces any trailing user line)
    """
    if not truth_statement:
        raise ValueError("truth_statement must be non-empty")
    if not false_presup_question:
        raise ValueError("false_presup_question must be non-empty")

    messages: list[ChatMessage] = [
        ChatMessage(role="user", content=f"{truth_prefix}{truth_statement}"),
        ChatMessage(role="assistant", content=_ACK_TEXT),
    ]
    for i, line in enumerate(_split_filler(filler_turns)):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(ChatMessage(role=role, content=line))

    if messages[-1].role == "assistant":
        messages.append(ChatMessage(role="user", content=false_presup_question))
    else:
        messages[-1] = ChatMessage(role="user", content=false_presup_question)
    return messages


def build_conversation_with_marker(
    truth_statement: str,
    filler_turns: str,
    false_presup_question: str,
    condition: str,
    *,
    reminder_every: int = 10,
) -> list[ChatMessage]:
    """Variant of :func:`build_conversation` for the marker-rescue ablation."""
    if condition not in MARKER_CONDITIONS:
        raise KeyError(
            f"Unknown marker condition {condition!r}. Valid: {sorted(MARKER_CONDITIONS)}"
        )

    if condition != "repeated_marker":
        prefix = MARKER_CONDITIONS[condition]["truth_prefix"]
        return build_conversation(
            truth_statement,
            filler_turns,
            false_presup_question,
            truth_prefix=prefix,
        )

    # repeated_marker: keep the standard "Just so you know: " establishment,
    # then re-inject a "[Reminder: <truth>]" prefix on user turns periodically.
    messages: list[ChatMessage] = [
        ChatMessage(role="user", content=f"Just so you know: {truth_statement}"),
        ChatMessage(role="assistant", content=_ACK_TEXT),
    ]
    for i, line in enumerate(_split_filler(filler_turns)):
        role = "user" if i % 2 == 0 else "assistant"
        if role == "user" and i > 0 and i % reminder_every == 0:
            line = f"[Reminder: {truth_statement}] {line}"
        messages.append(ChatMessage(role=role, content=line))

    if messages[-1].role == "assistant":
        messages.append(ChatMessage(role="user", content=false_presup_question))
    else:
        messages[-1] = ChatMessage(role="user", content=false_presup_question)
    return messages
