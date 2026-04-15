"""Shared structured JSON logging setup for pet-eval.

All CLI entry points and runners should call ``setup_logging()`` once at
startup to configure the root logger with a JSON formatter.
"""
from __future__ import annotations

import logging

from pythonjsonlogger import jsonlogger


def setup_logging() -> None:
    """Configure structured JSON logging for the root logger.

    Adds a :class:`~logging.StreamHandler` with a JSON formatter to the root
    logger and sets the level to ``INFO``.  Safe to call multiple times — will
    not duplicate handlers if the root logger already has a JSON handler.
    """
    root = logging.getLogger()

    # Avoid adding duplicate handlers if called more than once.
    for h in root.handlers:
        if isinstance(getattr(h, "formatter", None), jsonlogger.JsonFormatter):
            return

    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
