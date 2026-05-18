"""PDTB 3.0 → JSON.

Builds the two artefacts that Experiments 1 and 4–5 read:

* ``pdtb_train.json`` — full pool from sections 02–22, used for padding.
* ``pdtb_test_balanced.json`` — section 23, balanced to ``per_cell``
  examples per ``(rel_type, sense_l1)`` cell (8 cells × 50 = 400 by default).

Run via ``scripts/prepare_pdtb.py``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

from ..io_utils import write_json
from ..logging_utils import get_logger
from .pipe import PipeRelation, VALID_L1_SENSES, VALID_TYPES, parse_pipe_rows

_log = get_logger(__name__)

DEFAULT_TRAIN_SECTIONS: tuple[str, ...] = tuple(f"{i:02d}" for i in range(2, 22))
DEFAULT_TEST_SECTIONS: tuple[str, ...] = ("23",)


def _iter_documents(
    pdtb_root: Path,
    sections: Iterable[str],
    *,
    file_limit: int | None = None,
) -> Iterable[tuple[str, Path, Path]]:
    """Yield ``(doc_id, pipe_path, raw_path)`` for every document in ``sections``.

    ``file_limit`` (per section) is the smoke-test knob used by
    ``--smoke``: only the first N files in each section are visited.
    """
    gold_root = pdtb_root / "data" / "gold"
    raw_root = pdtb_root / "data" / "raw"
    for section in sections:
        gold_dir = gold_root / section
        if not gold_dir.is_dir():
            _log.warning("section missing: %s", gold_dir)
            continue
        files = sorted(p for p in gold_dir.iterdir() if p.is_file() and not p.name.startswith("."))
        if file_limit is not None:
            files = files[:file_limit]
        for pipe_path in files:
            raw_path = raw_root / section / pipe_path.name
            if not raw_path.is_file():
                continue
            yield pipe_path.name, pipe_path, raw_path


def _read_relations(
    pdtb_root: Path,
    sections: Iterable[str],
    *,
    file_limit: int | None = None,
) -> list[PipeRelation]:
    rels: list[PipeRelation] = []
    for doc_id, pipe_path, raw_path in _iter_documents(pdtb_root, sections, file_limit=file_limit):
        pipe_text = pipe_path.read_text(encoding="utf-8", errors="replace")
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
        rels.extend(parse_pipe_rows(pipe_text, raw_text, document_id=doc_id))
    return rels


def _to_record(rel: PipeRelation, *, section: str | None = None) -> dict[str, object]:
    record = {
        "rel_type": rel.rel_type,
        "sense_l1": rel.sense_l1,
        "sense_full": rel.sense_full,
        "arg1": rel.arg1,
        "arg2": rel.arg2,
        "document_id": rel.document_id,
    }
    if section is not None:
        record["section"] = section
    return record


def _balanced_sample(
    rels: list[PipeRelation],
    *,
    per_cell: int,
    seed: int,
) -> list[PipeRelation]:
    rng = random.Random(seed)
    by_cell: dict[tuple[str, str], list[PipeRelation]] = {}
    for r in rels:
        by_cell.setdefault((r.rel_type, r.sense_l1), []).append(r)

    out: list[PipeRelation] = []
    for rel_type in sorted(VALID_TYPES):
        for sense in sorted(VALID_L1_SENSES):
            pool = by_cell.get((rel_type, sense), [])
            if not pool:
                _log.warning("no examples for cell (%s, %s)", rel_type, sense)
                continue
            n = min(per_cell, len(pool))
            chosen = rng.sample(pool, n)
            if n < per_cell:
                _log.warning(
                    "cell (%s, %s) only had %d examples (wanted %d)",
                    rel_type,
                    sense,
                    n,
                    per_cell,
                )
            out.extend(chosen)
    return out


def build(
    *,
    pdtb_root: Path,
    out_dir: Path,
    train_sections: Iterable[str] = DEFAULT_TRAIN_SECTIONS,
    test_sections: Iterable[str] = DEFAULT_TEST_SECTIONS,
    per_cell: int = 50,
    seed: int = 42,
    smoke_files_per_section: int | None = None,
) -> dict[str, Path]:
    """Convert PDTB 3.0 → train + balanced-test JSON artefacts.

    Parameters
    ----------
    pdtb_root:
        Path to the directory that contains ``data/gold/`` and ``data/raw/``.
        For the LDC ISO this is ``<mount>/PDTB-3.0``.
    smoke_files_per_section:
        If set, only the first N files of each section are read. Used by
        ``--smoke`` to verify the pipeline before running on the whole corpus.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rels = _read_relations(
        pdtb_root, train_sections, file_limit=smoke_files_per_section
    )
    _log.info("train: %d valid relations", len(train_rels))
    train_path = out_dir / "pdtb_train.json"
    write_json(
        [_to_record(r) for r in train_rels],
        train_path,
    )

    test_rels = _read_relations(
        pdtb_root, test_sections, file_limit=smoke_files_per_section
    )
    _log.info("test (raw): %d valid relations", len(test_rels))
    balanced = _balanced_sample(test_rels, per_cell=per_cell, seed=seed)
    _log.info("test (balanced): %d relations", len(balanced))
    test_path = out_dir / "pdtb_test_balanced.json"
    write_json(
        [_to_record(r) for r in balanced],
        test_path,
    )
    return {"train": train_path, "test": test_path}
