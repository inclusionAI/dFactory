"""Run metrics rebuilt from events."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from anexus.events import EventEnvelope

_OUTPUT_FAILURE_EVENTS = {"attempt.output_publish.failed", "attempt.output_failed"}
_ARTIFACT_FAILURE_EVENTS = {
    "artifacts.failed",
    "artifacts.download_failed",
    "artifact.upload_failed",
}
_ERROR_EVENTS = {
    "attempt.retrying",
    "attempt.failed",
    "attempt.cancelled",
    *_OUTPUT_FAILURE_EVENTS,
    "cleanup.failed",
    *_ARTIFACT_FAILURE_EVENTS,
    "gateway.trace_export.failed",
}


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    events_seen: int = 0
    attempts_queued: int = 0
    attempts_admitted: int = 0
    attempts_started: int = 0
    attempts_retrying: int = 0
    attempts_failed: int = 0
    cleanup_pending: int = 0
    artifact_pending: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    error_kind_counts: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    queue_wait_seconds_total: float = 0.0
    global_wait_seconds_total: float = 0.0
    agent_wait_seconds_total: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "events_seen": self.events_seen,
            "attempts_queued": self.attempts_queued,
            "attempts_admitted": self.attempts_admitted,
            "attempts_started": self.attempts_started,
            "attempts_retrying": self.attempts_retrying,
            "attempts_failed": self.attempts_failed,
            "cleanup_pending": self.cleanup_pending,
            "artifact_pending": self.artifact_pending,
            "status_counts": dict(self.status_counts),
            "error_kind_counts": dict(self.error_kind_counts),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "queue_wait_seconds_total": self.queue_wait_seconds_total,
            "global_wait_seconds_total": self.global_wait_seconds_total,
            "agent_wait_seconds_total": self.agent_wait_seconds_total,
        }


def reduce_run_metrics(events: Iterable[EventEnvelope | Mapping[str, Any]]) -> RunMetrics:
    events_seen = 0
    run_id = ""
    attempts_queued = 0
    attempts_admitted = 0
    attempts_started: set[tuple[str, str, str, str]] = set()
    attempts_retrying = 0
    attempts_failed: set[tuple[str, str, str, str]] = set()
    cleanup_pending: set[tuple[str, str, str, str]] = set()
    artifact_pending: set[tuple[str, str, str, str]] = set()
    terminal_status: dict[tuple[str, str, str, str], str] = {}
    error_kind_counts: Counter[str] = Counter()
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    queue_wait_seconds_total = 0.0
    global_wait_seconds_total = 0.0
    agent_wait_seconds_total = 0.0

    for event in events:
        data = event.to_dict() if isinstance(event, EventEnvelope) else dict(event)
        events_seen += 1
        context = dict(data.get("context") or {})
        payload = dict(data.get("payload") or {})
        name = str(data.get("name", ""))
        attempt_id = _attempt_identity(context)
        if not run_id:
            run_id = _optional_str(context.get("run_id")) or ""
        for error_kind in _error_kinds_from_payload(name, payload):
            error_kind_counts[error_kind] += 1
        if name == "attempt.queued":
            attempts_queued += 1
        elif name == "attempt.admitted":
            attempts_admitted += 1
            queue_wait_seconds_total += _optional_float(payload.get("queue_wait_seconds")) or 0.0
            global_wait_seconds_total += _optional_float(payload.get("global_wait_seconds")) or 0.0
            agent_wait_seconds_total += _optional_float(payload.get("agent_wait_seconds")) or 0.0
        if attempt_id is None:
            continue

        if name == "attempt.started":
            attempts_started.add(attempt_id)
        elif name == "attempt.retrying":
            attempts_retrying += 1
        elif name in {"attempt.failed", "attempt.cancelled"}:
            attempts_failed.add(attempt_id)
            terminal_status[attempt_id] = _status_from_error_payload(payload)
        elif name in _OUTPUT_FAILURE_EVENTS:
            attempts_failed.add(attempt_id)
            terminal_status[attempt_id] = "failed"
        elif name == "cleanup.failed":
            cleanup_pending.add(attempt_id)
        elif name in _ARTIFACT_FAILURE_EVENTS:
            artifact_pending.add(attempt_id)
        elif name == "llm.completed":
            usage = dict(payload.get("usage") or {})
            input_tokens += int(usage.get("input_tokens", 0))
            output_tokens += int(usage.get("output_tokens", 0))
            total_tokens += int(usage.get("total_tokens", 0))
        elif name == "attempt.output_published":
            status = _optional_str(payload.get("status"))
            if status:
                terminal_status[attempt_id] = status

    status_counter = Counter(terminal_status.values())
    return RunMetrics(
        run_id=run_id,
        events_seen=events_seen,
        attempts_queued=attempts_queued,
        attempts_admitted=attempts_admitted,
        attempts_started=len(attempts_started),
        attempts_retrying=attempts_retrying,
        attempts_failed=len(attempts_failed),
        cleanup_pending=len(cleanup_pending),
        artifact_pending=len(artifact_pending),
        status_counts=dict(status_counter),
        error_kind_counts=dict(error_kind_counts),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        queue_wait_seconds_total=queue_wait_seconds_total,
        global_wait_seconds_total=global_wait_seconds_total,
        agent_wait_seconds_total=agent_wait_seconds_total,
    )


def _status_from_error_payload(payload: dict[str, Any]) -> str:
    status = _optional_str(payload.get("status"))
    if status in {"cancelled", "timeout", "failed"}:
        return status
    if payload.get("kind") == "cancelled":
        return "cancelled"
    if payload.get("kind") == "timeout":
        return "timeout"
    return "failed"


def _error_kinds_from_payload(name: str, payload: dict[str, Any]) -> list[str]:
    if name not in _ERROR_EVENTS:
        return []
    kinds: list[str] = []
    kind = _optional_str(payload.get("kind"))
    if kind:
        kinds.append(kind)
    errors = payload.get("errors")
    if isinstance(errors, list):
        for error in errors:
            if isinstance(error, Mapping):
                nested_kind = _optional_str(error.get("kind"))
                if nested_kind:
                    kinds.append(nested_kind)
    return kinds


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _attempt_identity(
    context: Mapping[str, Any],
) -> tuple[str, str, str, str] | None:
    attempt_id = _optional_str(context.get("attempt_id"))
    if attempt_id is None:
        return None
    return (
        _optional_str(context.get("run_id")) or "",
        _optional_str(context.get("agent_name")) or "",
        _optional_str(context.get("task_id")) or "",
        attempt_id,
    )
