"""Context-local fields for structured logs."""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from anexus.core.context import TaskContext

LOG_CONTEXT_FIELDS = frozenset(
    {
        "run_id",
        "agent_name",
        "task_id",
        "version",
        "attempt_id",
        "agent_spec_name",
        "gateway_session_id",
        "sandbox_id",
        "worker_pid",
        "process_id",
        "episode_id",
        "swarm_id",
        "node_id",
        "phase",
        "reason",
        "status",
        "provider",
        "operation",
        "resource_name",
        "resource_role",
        "object_id",
    }
)

_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "anexus_log_context",
    default=None,
)


class LogContextFilter(logging.Filter):
    """Inject contextvars-backed identity fields into Anexus log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        bound_context = get_log_context()
        if not bound_context:
            return True
        explicit_context = _normalize_context(getattr(record, "anexus_context", None))
        merged = {**bound_context, **explicit_context}
        if merged:
            record.anexus_context = merged
        return True


def get_log_context() -> dict[str, Any]:
    """Return a copy of the current logging context."""

    return dict(_LOG_CONTEXT.get() or {})


@contextmanager
def bind_log_context(
    context: TaskContext | Mapping[str, object] | None = None,
    **fields: object,
) -> Iterator[dict[str, Any]]:
    """Temporarily bind structured fields to logs emitted in this context."""

    current = get_log_context()
    updates: dict[str, Any] = {}
    for source in (_normalize_context(context), fields):
        for key, value in source.items():
            if key in LOG_CONTEXT_FIELDS and value not in ("", None):
                updates[key] = value
    token = _LOG_CONTEXT.set({**current, **updates})
    try:
        yield get_log_context()
    finally:
        _LOG_CONTEXT.reset(token)


def _normalize_context(context: object) -> dict[str, Any]:
    if isinstance(context, TaskContext):
        return context.to_fields()
    if isinstance(context, Mapping):
        return {str(key): value for key, value in context.items()}
    return {}
