"""Logging helpers. Plain stdlib logging; pluggable if we ever want JSON."""

from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def get_logger(name: str = "trend_engine", level: str = "INFO") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )
        root = logging.getLogger("trend_engine")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)
