"""Loader for pre-processed TED-MDB records (Experiment 4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..io_utils import read_json

_REQUIRED = ("language", "rel_type", "arg1", "arg2", "sense_l1")
_VALID_TYPES = {"Implicit", "Explicit"}


@dataclass(frozen=True)
class TedMdbExample:
    language: str
    rel_type: str
    sense_l1: str
    arg1: str
    arg2: str
    extras: dict[str, object]

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "TedMdbExample":
        for key in _REQUIRED:
            if key not in raw:
                raise ValueError(f"TED-MDB record missing key {key!r}: {sorted(raw)}")
        rel_type = str(raw["rel_type"])
        if rel_type not in _VALID_TYPES:
            raise ValueError(f"Unexpected rel_type={rel_type!r}")
        return cls(
            language=str(raw["language"]),
            rel_type=rel_type,
            sense_l1=str(raw["sense_l1"]),
            arg1=str(raw["arg1"]),
            arg2=str(raw["arg2"]),
            extras={k: v for k, v in raw.items() if k not in _REQUIRED},
        )


def load_ted_mdb(path: str | Path) -> list[TedMdbExample]:
    return [TedMdbExample.from_dict(r) for r in read_json(path)]
