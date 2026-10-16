"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anexus.core.context import TaskContext
from anexus.core.errors import ErrorEnvelope
from anexus.core.files import (
    JsonlReadError,
    read_jsonl_objects,
    read_jsonl_objects_recoverable,
)
from anexus.events import EventEnvelope
from anexus.logs.context import LogContextFilter


class EventLogSink:
    """Event bus sink that mirrors runtime facts into structured logs."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        level: int = logging.INFO,
    ):
        self.logger = logger or logging.getLogger("anexus.events")
        self.level = level

    def emit(self, event: EventEnvelope) -> None:
        self.logger.log(
            self.level,
            event.name,
            extra=log_extra(
                context=event.context,
                fields={
                    "event_id": event.event_id,
                    "event_name": event.name,
                    "event_timestamp": event.timestamp,
                    "payload": event.payload,
                },
            ),
        )


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            "process_id": record.process,
            "process_name": record.processName,
            "thread_name": record.threadName,
        }
        context = _normalize_context(getattr(record, "anexus_context", None))
        if context:
            payload["context"] = context
        error = _normalize_error(getattr(record, "error_envelope", None))
        if error:
            payload["error"] = error
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            payload["fields"] = extra_fields
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def log_extra(
    *,
    context: TaskContext | dict[str, Any] | None = None,
    error: ErrorEnvelope | dict[str, Any] | None = None,
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build logging `extra` fields with stable Anexus keys."""

    extra: dict[str, Any] = {}
    if context is not None:
        extra["anexus_context"] = context
    if error is not None:
        extra["error_envelope"] = error
    if fields is not None:
        extra["extra_fields"] = fields
    return extra


def make_json_log_handler(
    stream: Any | None = None,
    *,
    level: int | str = logging.NOTSET,
) -> logging.Handler:
    """Create a stream handler that writes one JSON log object per record."""

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.setLevel(level)
    handler.addFilter(LogContextFilter())
    return handler


def configure_json_logging(
    logger_name: str = "anexus",
    *,
    stream: Any | None = None,
    level: int | str = logging.INFO,
    replace_handlers: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """Configure a logger subtree for structured Anexus logs."""

    logger = logging.getLogger(logger_name)
    handler = make_json_log_handler(stream, level=level)
    if replace_handlers:
        logger.handlers = [handler]
    else:
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = propagate
    return logger


def read_json_log_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read structured Anexus JSONL log rows from disk."""

    return read_jsonl_objects(_require_log_path(path))


def read_json_log_rows_recoverable(
    path: str | Path,
) -> tuple[list[dict[str, Any]], list[JsonlReadError]]:
    """Read valid structured log rows and report invalid JSONL rows."""

    return read_jsonl_objects_recoverable(_require_log_path(path))


def _require_log_path(path: str | Path) -> Path:
    log_path = Path(path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    return log_path


def _normalize_context(context: object) -> dict[str, Any]:
    if isinstance(context, TaskContext):
        return context.to_fields()
    if isinstance(context, dict):
        return context
    return {}


def _normalize_error(error: object) -> dict[str, Any]:
    if isinstance(error, ErrorEnvelope):
        return error.to_dict()
    if isinstance(error, dict):
        return error
    return {}
