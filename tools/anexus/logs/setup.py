"""Logging setup helpers."""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from anexus.logs.context import LogContextFilter
from anexus.logs.json import JsonFormatter

_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(
    logger_name: str = "anexus",
    *,
    stream: Any | None = None,
    level: int | str = logging.INFO,
    json_output: bool = True,
    replace_handlers: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """Configure and return an Anexus logger."""

    logger = logging.getLogger(logger_name)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
    handler.addFilter(LogContextFilter())
    if replace_handlers:
        for existing in list(logger.handlers):
            logger.removeHandler(existing)
            existing.close()
        logger.addHandler(handler)
    else:
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = propagate
    return logger


@contextlib.contextmanager
def run_file_logging(path: str | Path) -> Iterator[logging.Logger]:
    """Route Anexus diagnostics and third-party warnings to one run log."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_TEXT_FORMAT))

    root = logging.getLogger()
    anexus = logging.getLogger("anexus")
    previous_root_handlers = list(root.handlers)
    previous_root_level = root.level
    previous_anexus_handlers = list(anexus.handlers)
    previous_anexus_level = anexus.level
    previous_anexus_propagate = anexus.propagate
    root.handlers = [handler]
    root.setLevel(logging.WARNING)
    anexus.handlers = []
    anexus.setLevel(logging.DEBUG)
    anexus.propagate = True
    logging.captureWarnings(True)
    try:
        yield anexus
    finally:
        logging.captureWarnings(False)
        root.handlers = previous_root_handlers
        root.setLevel(previous_root_level)
        anexus.handlers = previous_anexus_handlers
        anexus.setLevel(previous_anexus_level)
        anexus.propagate = previous_anexus_propagate
        handler.close()
