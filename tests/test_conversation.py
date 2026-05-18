"""Tests for the multi-turn conversation builders."""

from __future__ import annotations

import pytest

from long_context_discourse.conversation import (
    build_conversation,
    build_conversation_with_marker,
)


# Two-line filler: roles after the truth+ack become user, assistant — so the
# false-presup question is appended at the end and no filler content is
# replaced. Tests below depend on that layout.
_FILLER = """A: hello
B: hi"""


def test_build_conversation_basic_structure() -> None:
    msgs = build_conversation(
        truth_statement="John has never owned a car.",
        filler_turns=_FILLER,
        false_presup_question="What color is John's car?",
    )
    assert msgs[0].role == "user"
    assert "John has never owned a car." in msgs[0].content
    assert msgs[1].role == "assistant"
    # Last message must always be the false-presup user question.
    assert msgs[-1].role == "user"
    assert msgs[-1].content == "What color is John's car?"


def test_build_conversation_strips_speaker_prefixes() -> None:
    msgs = build_conversation("X is true.", _FILLER, "Q?")
    contents = [m.content for m in msgs]
    assert "hello" in contents and "hi" in contents


def test_build_conversation_replaces_trailing_user_filler() -> None:
    # Three-line filler ends on a user turn; the question must replace it
    # so no two consecutive user messages appear.
    filler = "A: a-one\nB: b-one\nA: a-two-trailing"
    msgs = build_conversation("T", filler, "Q?")
    assert msgs[-1].role == "user" and msgs[-1].content == "Q?"
    assert all(m.content != "a-two-trailing" for m in msgs)


def test_build_conversation_validates_inputs() -> None:
    with pytest.raises(ValueError):
        build_conversation("", "filler", "Q?")
    with pytest.raises(ValueError):
        build_conversation("truth", "filler", "")


def test_marker_condition_no_marker_omits_prefix() -> None:
    msgs = build_conversation_with_marker("T", _FILLER, "Q?", "no_marker")
    assert msgs[0].content == "T"


def test_marker_condition_strong_marker_prepended() -> None:
    msgs = build_conversation_with_marker("T", _FILLER, "Q?", "strong_marker")
    assert msgs[0].content.startswith("IMPORTANT")


def test_marker_condition_repeated_injects_reminder() -> None:
    long_filler = "\n".join(["A: line"] * 30)
    msgs = build_conversation_with_marker(
        "T", long_filler, "Q?", "repeated_marker", reminder_every=10
    )
    assert any("[Reminder: T]" in m.content for m in msgs)


def test_marker_condition_unknown_raises() -> None:
    with pytest.raises(KeyError):
        build_conversation_with_marker("T", "A: x", "Q?", "nope")
