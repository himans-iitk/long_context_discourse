"""GUM RST → JSON for the Experiment 3 probe.

For each ``.rs4`` file we emit a document with ``segments``: a list of
``{text, role, position_ratio, genre}`` records. Role assignment follows
the standard rs3/rs4 convention:

* a segment whose ``relname == "span"`` is the **nucleus** of an RST relation;
* a segment whose parent is a ``multinuc`` group is also a **nucleus**;
* otherwise the segment is a **satellite**.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..io_utils import write_json
from ..logging_utils import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class GumDoc:
    document_id: str
    genre: str
    segments: list[dict[str, object]]


def _genre_from_filename(name: str) -> str:
    # Filenames look like ``GUM_news_iraq.rs4`` or ``GENTLE_dictionary_next.rs4``.
    parts = name.replace(".rs4", "").split("_")
    return parts[1] if len(parts) >= 2 else ""


def parse_rs4(path: Path) -> GumDoc:
    """Parse one .rs4 file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    root = ET.fromstring(text)
    body = root.find("body")
    if body is None:
        return GumDoc(
            document_id=path.stem,
            genre=_genre_from_filename(path.name),
            segments=[],
        )

    # Build the group table once: id → group element type.
    group_type: dict[str, str] = {}
    for grp in body.findall("group"):
        gid = grp.get("id")
        gtype = grp.get("type", "span")
        if gid:
            group_type[gid] = gtype

    raw_segments: list[tuple[int, str, str]] = []  # (id_int, role, text)
    for seg in body.findall("segment"):
        sid = seg.get("id")
        if not sid:
            continue
        try:
            sid_int = int(sid)
        except ValueError:
            continue
        relname = (seg.get("relname") or "").strip()
        parent = seg.get("parent") or ""
        if relname == "span":
            role = "nucleus"
        elif group_type.get(parent) == "multinuc":
            role = "nucleus"
        else:
            role = "satellite"
        text_value = (seg.text or "").strip()
        if not text_value:
            continue
        raw_segments.append((sid_int, role, text_value))

    if not raw_segments:
        return GumDoc(
            document_id=path.stem,
            genre=_genre_from_filename(path.name),
            segments=[],
        )

    raw_segments.sort(key=lambda r: r[0])
    n = len(raw_segments)
    out_segments: list[dict[str, object]] = []
    for rank, (sid_int, role, text_value) in enumerate(raw_segments):
        position_ratio = rank / max(n - 1, 1)
        out_segments.append(
            {
                "segment_id": sid_int,
                "text": text_value,
                "role": role,
                "position_ratio": round(position_ratio, 6),
            }
        )
    return GumDoc(
        document_id=path.stem,
        genre=_genre_from_filename(path.name),
        segments=out_segments,
    )


def build(
    *,
    rs4_dir: Path,
    out_path: Path,
    file_limit: int | None = None,
) -> Path:
    """Walk ``rs4_dir`` and write the probing JSON to ``out_path``."""
    rs4_dir = Path(rs4_dir).expanduser()
    files = sorted(p for p in rs4_dir.iterdir() if p.suffix == ".rs4")
    if file_limit is not None:
        files = files[:file_limit]
    _log.info("processing %d .rs4 files from %s", len(files), rs4_dir)

    docs: list[dict[str, object]] = []
    n_segments = 0
    for path in files:
        doc = parse_rs4(path)
        if not doc.segments:
            continue
        docs.append(
            {
                "document_id": doc.document_id,
                "genre": doc.genre,
                "segments": doc.segments,
            }
        )
        n_segments += len(doc.segments)
    _log.info("%d documents, %d total segments", len(docs), n_segments)

    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(docs, out_path)
    return out_path
