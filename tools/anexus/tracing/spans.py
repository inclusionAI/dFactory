"""Span reconstruction from event streams."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from anexus.events import EventEnvelope


@dataclass(frozen=True)
class TraceSpan:
    trace_id: str
    span_id: str
    name: str
    run_id: str
    agent_name: str
    task_id: str
    attempt_id: str
    start_time: str
    end_time: str | None = None
    status: str = "UNSET"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_otel_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "start_time_unix_nano": _unix_nano(self.start_time),
            "end_time_unix_nano": _unix_nano(self.end_time) if self.end_time else None,
            "attributes": self.attributes,
            "events": self.events,
            "status": {"code": self.status},
        }


class TraceRecorder:
    """Event sink that records one span per task attempt."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._spans: dict[str, TraceSpan] = {}

    def emit(self, event: EventEnvelope) -> None:
        with self._lock:
            self._spans = _apply_event(self._spans, event.to_dict())

    def spans(self) -> list[TraceSpan]:
        with self._lock:
            return list(self._spans.values())


def replay_spans(events: Iterable[EventEnvelope | Mapping[str, Any]]) -> list[TraceSpan]:
    spans: dict[str, TraceSpan] = {}
    for event in events:
        data = event.to_dict() if isinstance(event, EventEnvelope) else dict(event)
        spans = _apply_event(spans, data)
    return list(spans.values())


def _apply_event(spans: dict[str, TraceSpan], event: dict[str, Any]) -> dict[str, TraceSpan]:
    context = dict(event.get("context") or {})
    attempt_id = _optional_str(context.get("attempt_id"))
    run_id = _optional_str(context.get("run_id"))
    agent_name = _optional_str(context.get("agent_name")) or ""
    task_id = _optional_str(context.get("task_id"))
    if not attempt_id or not run_id or not task_id:
        return spans

    name = str(event.get("name", ""))
    timestamp = str(event.get("timestamp") or datetime.now(UTC).isoformat())
    span_key = _attempt_key(
        run_id=run_id,
        agent_name=agent_name,
        task_id=task_id,
        attempt_id=attempt_id,
    )
    span = spans.get(span_key)
    if span is None:
        span = TraceSpan(
            trace_id=span_key,
            span_id=span_key,
            name="task.attempt",
            run_id=run_id,
            agent_name=agent_name,
            task_id=task_id,
            attempt_id=attempt_id,
            start_time=timestamp,
            attributes=_attributes_from_context(context),
        )

    events = [
        *span.events,
        {
            "name": name,
            "timestamp": timestamp,
            "attributes": dict(event.get("payload") or {}),
        },
    ]
    end_time = span.end_time
    status = span.status
    if name in {"attempt.failed", "attempt.output_publish.failed"}:
        end_time = timestamp
        status = "ERROR"
    elif name == "attempt.output_published":
        end_time = timestamp
        status = _span_status(str(dict(event.get("payload") or {}).get("status", "")))

    next_spans = dict(spans)
    next_spans[span_key] = TraceSpan(
        trace_id=span.trace_id,
        span_id=span.span_id,
        name=span.name,
        run_id=span.run_id,
        agent_name=span.agent_name,
        task_id=span.task_id,
        attempt_id=span.attempt_id,
        start_time=span.start_time,
        end_time=end_time,
        status=status,
        attributes=span.attributes,
        events=events,
    )
    return next_spans


def _attempt_key(*, run_id: str, agent_name: str, task_id: str, attempt_id: str) -> str:
    return json.dumps(
        [run_id, agent_name, task_id, attempt_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _attributes_from_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in context.items()
        if value is not None
        and key
        in {
            "run_id",
            "task_id",
            "version",
            "attempt_id",
            "agent_name",
            "agent_spec_name",
            "gateway_session_id",
            "sandbox_id",
            "worker_pid",
        }
    }


def _span_status(status: str) -> str:
    if status == "completed":
        return "OK"
    if status:
        return "ERROR"
    return "UNSET"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _unix_nano(value: str) -> int:
    return int(_parse_time(value).timestamp() * 1_000_000_000)
