"""Loader for pre-processed GUM RST segments (Experiment 3).

We expect upstream pre-processing (in your data-prep pipeline) to emit a
list of documents, each with a ``segments`` list. Every segment carries
``text``, ``role`` (``nucleus`` / ``satellite``) and ``position_ratio`` ∈
[0, 1] giving its location in the source document.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..io_utils import read_json


@dataclass(frozen=True)
class GumSegment:
    text: str
    role: str  # "nucleus" or "satellite"
    position_ratio: float
    genre: str
    document_id: str

    @property
    def label(self) -> int:
        return 1 if self.role == "nucleus" else 0


def load_gum_segments(
    path: str | Path,
    *,
    min_chars: int = 20,
    max_chars: int = 500,
) -> list[GumSegment]:
    raw = read_json(path)
    segments: list[GumSegment] = []
    for doc in raw:
        doc_id = str(doc.get("document_id") or doc.get("id") or "")
        genre = str(doc.get("genre") or "")
        for seg in doc.get("segments", []):
            text = (seg.get("text") or "").strip()
            role = seg.get("role")
            pos = seg.get("position_ratio")
            if not text or role not in {"nucleus", "satellite"} or pos is None:
                continue
            if not (min_chars <= len(text) <= max_chars):
                continue
            segments.append(
                GumSegment(
                    text=text,
                    role=str(role),
                    position_ratio=float(pos),
                    genre=genre,
                    document_id=doc_id,
                )
            )
    return segments
