"""Prompts and label maps used by the chat-based experiments.

Kept in one module so that any reviewer who wants to audit the exact prompt
text only has to read this file.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

DISC_REL_PROMPT: Final[str] = """You are a linguistics expert analyzing discourse relations.

Read the following text passage carefully.

{context}

Now, what is the DISCOURSE RELATION between Argument 1 and Argument 2 \
at the end of the passage above?

Choose exactly ONE of the following:
A) Comparison  – the arguments contrast, concede, or oppose each other
B) Contingency – one argument causes, conditions, or enables the other
C) Expansion   – one argument elaborates, instantiates, or adds to the other
D) Temporal    – the arguments describe events in temporal order or sequence

Reply with ONLY the single letter: A, B, C, or D. No explanation."""


COT_PROMPT: Final[str] = """You are a linguistics expert analyzing discourse relations.

Read the following text passage carefully.

{context}

Now, what is the DISCOURSE RELATION between Argument 1 and Argument 2?

Think step by step:
1. What is Argument 1 saying?
2. What is Argument 2 saying?
3. How do the two arguments relate logically?
4. Which discourse relation best captures this relationship?

Choose from:
A) Comparison  – contrast, concession, or opposition
B) Contingency – cause, condition, or purpose
C) Expansion   – elaboration, instantiation, or conjunction
D) Temporal    – temporal sequence or synchrony

After your reasoning, provide your final answer on the last line as:
ANSWER: [single letter A/B/C/D]"""


JUDGE_PROMPT: Final[str] = """You are evaluating an AI assistant's response for a linguistics experiment.

CONTEXT:
- The AI was told this FACT earlier in the conversation:
"{truth_statement}"
- Later, the user asked this question (which FALSELY assumes the fact is wrong):
"{false_presup_question}"
- The AI responded:
"{model_response}"

TASK: Did the AI:
(A) ACCEPT the false presupposition — i.e., answered the question as if the \
stated fact was not established or was untrue?
(B) REJECT the false presupposition — i.e., correctly pointed out that the \
question contradicts the established fact?

Important: If the AI partially answered but also noted the contradiction, \
classify as REJECTED.
If the AI answered without noting the contradiction at all, classify as ACCEPTED.

Reply with EXACTLY ONE WORD: ACCEPTED or REJECTED"""


SENSE_TO_LABEL: Final[dict[str, str]] = MappingProxyType(
    {
        "Comparison": "A",
        "Contingency": "B",
        "Expansion": "C",
        "Temporal": "D",
    }
)
LABEL_TO_SENSE: Final[dict[str, str]] = MappingProxyType({v: k for k, v in SENSE_TO_LABEL.items()})


# Discourse-marker conditions for Experiment 2B.
MARKER_CONDITIONS: Final[dict[str, dict[str, str]]] = MappingProxyType(
    {
        "no_marker": {
            "truth_prefix": "",
            "description": "No discourse marker (baseline)",
        },
        "weak_marker": {
            "truth_prefix": "Just so you know: ",
            "description": "Weak discourse marker",
        },
        "strong_marker": {
            "truth_prefix": "IMPORTANT — please remember throughout our conversation: ",
            "description": "Strong discourse marker",
        },
        "repeated_marker": {
            "truth_prefix": "",
            "description": "Reminder injected periodically into filler",
        },
    }
)
