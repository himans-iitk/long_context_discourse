"""Raw-corpus → JSON converters for Experiments 1, 3, 4, 5.

Each submodule exposes a single ``build(...)`` function that returns the
exact record shape consumed by ``long_context_discourse.data.*`` loaders.
Thin CLI wrappers live under ``scripts/``.
"""

from __future__ import annotations
