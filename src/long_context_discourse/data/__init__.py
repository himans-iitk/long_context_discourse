"""Dataset loaders.

Each module exposes a small dataclass for one record plus a ``load_*``
function that returns ``list[record]``.
"""

from __future__ import annotations

from .gum import GumSegment, load_gum_segments
from .pdtb import PdtbExample, load_pdtb_balanced, load_pdtb_train
from .presupposition import PresupExample, load_presupposition_dataset
from .ted_mdb import TedMdbExample, load_ted_mdb

__all__ = [
    "GumSegment",
    "PdtbExample",
    "PresupExample",
    "TedMdbExample",
    "load_gum_segments",
    "load_pdtb_balanced",
    "load_pdtb_train",
    "load_presupposition_dataset",
    "load_ted_mdb",
]
