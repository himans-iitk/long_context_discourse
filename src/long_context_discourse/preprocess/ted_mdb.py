"""TED-MDB → JSON for Experiment 4 (cross-lingual).

The annotation files use the same pipe layout as PDTB-3 (the corpus was
designed to be PDTB-style). Raw text is one ``talk_NNNN_<lang>.txt`` per
talk, with character offsets aligned to that file.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..io_utils import write_json
from ..logging_utils import get_logger
from .pipe import parse_pipe_rows

_log = get_logger(__name__)

DEFAULT_LANGUAGES: tuple[str, ...] = (
    "English",
    "German",
    "Polish",
    "Russian",
    "Portuguese",
    "Turkish",
)


def _ann_files(lang_dir: Path) -> list[Path]:
    """All ``.txt`` annotation files under ``<lang>/ann/01``."""
    ann_dir = lang_dir / "ann" / "01"
    if not ann_dir.is_dir():
        return []
    return sorted(p for p in ann_dir.iterdir() if p.suffix == ".txt")


def _raw_for(ann_path: Path, lang_dir: Path) -> Path:
    """The raw text file matching an annotation filename."""
    return lang_dir / "raw" / "01" / ann_path.name


def _to_record(rel, language: str, talk_id: str) -> dict[str, object]:
    return {
        "language": language,
        "rel_type": rel.rel_type,
        "sense_l1": rel.sense_l1,
        "sense_full": rel.sense_full,
        "arg1": rel.arg1,
        "arg2": rel.arg2,
        "talk_id": talk_id,
    }


def build(
    *,
    ted_root: Path,
    out_path: Path,
    languages: Iterable[str] = DEFAULT_LANGUAGES,
    file_limit_per_lang: int | None = None,
) -> Path:
    ted_root = Path(ted_root).expanduser()
    out: list[dict[str, object]] = []

    for language in languages:
        lang_dir = ted_root / language
        if not lang_dir.is_dir():
            _log.warning("language directory missing: %s", lang_dir)
            continue
        ann_files = _ann_files(lang_dir)
        if file_limit_per_lang is not None:
            ann_files = ann_files[:file_limit_per_lang]
        kept = 0
        for ann_path in ann_files:
            raw_path = _raw_for(ann_path, lang_dir)
            if not raw_path.is_file():
                _log.warning("missing raw for %s", ann_path)
                continue
            pipe_text = ann_path.read_text(encoding="utf-8", errors="replace")
            raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
            rels = parse_pipe_rows(
                pipe_text, raw_text, document_id=ann_path.stem
            )
            for rel in rels:
                out.append(_to_record(rel, language=language, talk_id=ann_path.stem))
                kept += 1
        _log.info("%-12s %4d relations from %d files", language, kept, len(ann_files))

    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, out_path)
    return out_path
