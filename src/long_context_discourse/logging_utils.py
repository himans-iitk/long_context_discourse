"""Lightweight logging helper for library and scripts.

Library code obtains a named logger via :func:`get_logger`. Scripts should call
:func:`configure_logging` once at startup; importing this module never
configures handlers as a side-effect.
"""

from __future__ import annotations

import logging
import os
from typing import Final

_DEFAULT_FORMAT: Final[str] = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int | str | None = None) -> None:
    """Configure the root logger once.

    The level is resolved in this order: explicit ``level`` argument, the
    ``LCD_LOG_LEVEL`` environment variable, then ``INFO``.
    """
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
    elif level is None:
        env = os.environ.get("LCD_LOG_LEVEL", "INFO").upper()
        resolved = logging.getLevelName(env)
    else:
        resolved = level

    if not isinstance(resolved, int):
        resolved = logging.INFO

    logging.basicConfig(
        level=resolved,
        format=_DEFAULT_FORMAT,
        datefmt=_DATEFMT,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Prefer ``get_logger(__name__)`` over ``logging.getLogger(__name__)`` so
    that any future global configuration (e.g. structured logging) applies
    consistently.
    """
    return logging.getLogger(name)
