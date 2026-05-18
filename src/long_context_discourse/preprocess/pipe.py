"""Parser for the PDTB-3 / TED-MDB pipe annotation format.

Both corpora use the same column layout. We only consume the columns
needed by Experiments 1 and 4:

==== ============================================================
col   meaning
==== ============================================================
0     relation_type           (Explicit | Implicit | AltLex | …)
8     sense (dot-separated; first segment is the Level-1 sense)
14    arg1 char-span list     ("a..b" or "a..b;c..d" if discontinuous)
20    arg2 char-span list
==== ============================================================

Char offsets are **end-exclusive**, so ``raw[start:end]`` is the correct
Python slice. Discontinuous spans are joined with a single space when
materialised back into argument text — this matches the standard
practice in PDTB tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

VALID_TYPES: Final[frozenset[str]] = frozenset({"Implicit", "Explicit"})
VALID_L1_SENSES: Final[frozenset[str]] = frozenset(
    {"Comparison", "Contingency", "Expansion", "Temporal"}
)


@dataclass(frozen=True)
class PipeRelation:
    """One row of the pipe file with its arguments resolved against raw text."""

    rel_type: str
    sense_l1: str
    sense_full: str
    arg1: str
    arg2: str
    arg1_spans: tuple[tuple[int, int], ...]
    arg2_spans: tuple[tuple[int, int], ...]
    document_id: str


def _parse_span_list(raw: str) -> tuple[tuple[int, int], ...]:
    """Parse ``"9..35"`` or ``"9..35;195..239"`` into ``((9, 35), …)``.

    Returns an empty tuple if ``raw`` is empty or malformed.
    """
    raw = raw.strip()
    if not raw:
        return ()
    out: list[tuple[int, int]] = []
    for part in raw.split(";"):
        part = part.strip()
        if ".." not in part:
            continue
        try:
            lo_s, hi_s = part.split("..", 1)
            lo, hi = int(lo_s), int(hi_s)
        except ValueError:
            continue
        if hi > lo:
            out.append((lo, hi))
    return tuple(out)


def _slice(raw_text: str, spans: tuple[tuple[int, int], ...]) -> str:
    if not spans:
        return ""
    pieces = [raw_text[lo:hi] for lo, hi in spans]
    return " ".join(p.strip() for p in pieces if p.strip())


def parse_pipe_rows(
    pipe_text: str,
    raw_text: str,
    *,
    document_id: str,
) -> list[PipeRelation]:
    """Parse one pipe file. Discards rows without all four required fields.

    A row is kept iff:

    * ``rel_type`` is in :data:`VALID_TYPES`
    * level-1 sense is in :data:`VALID_L1_SENSES`
    * both ``arg1`` and ``arg2`` resolve to non-empty strings
    """
    out: list[PipeRelation] = []
    for line in pipe_text.splitlines():
        if not line:
            continue
        cols = line.split("|")
        if len(cols) < 21:
            continue
        rel_type = cols[0].strip()
        if rel_type not in VALID_TYPES:
            continue
        sense_full = cols[8].strip()
        if not sense_full:
            continue
        sense_l1 = sense_full.split(".", 1)[0]
        if sense_l1 not in VALID_L1_SENSES:
            continue
        arg1_spans = _parse_span_list(cols[14])
        arg2_spans = _parse_span_list(cols[20])
        arg1 = _slice(raw_text, arg1_spans)
        arg2 = _slice(raw_text, arg2_spans)
        if not arg1 or not arg2:
            continue
        out.append(
            PipeRelation(
                rel_type=rel_type,
                sense_l1=sense_l1,
                sense_full=sense_full,
                arg1=arg1,
                arg2=arg2,
                arg1_spans=arg1_spans,
                arg2_spans=arg2_spans,
                document_id=document_id,
            )
        )
    return out
