"""Long-context discourse experiments.

Reference implementation of the five experiments described in
"Coherence at Scale" (EMNLP 2026 submission).

Public surface is intentionally small; consumers should import from the
submodules they need (``from long_context_discourse.models import OpenRouterClient``).
"""

from __future__ import annotations

from ._version import __version__

__all__ = ["__version__"]
