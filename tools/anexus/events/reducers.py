"""Reducers that rebuild UI state from event replay."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from anexus.events.envelope import EventEnvelope

_STATUS_EVENTS = {
    "attempt.started": "started",
    "gateway.opened": "gateway_open",
    "gateway.ready": "gateway_open",
    "sandbox.created": "sandbox_ready",
    "fixtures.ready": "fixtures_ready",
    "llm.started": "llm_waiting",
    "llm.completed": "running",
    "agent.started": "running",
    "agent.completed": "evaluating",
    "artifacts.committed": "cleanup",
}
_SIDECAR_EVENTS = {
    "sidecar.created": "created",
    "sidecar.ready": "ready",
    "sidecar.deleted": "deleted",
}
_ARTIFACT_FAILURE_EVENTS = {"artifacts.failed", "artifact.upload_failed"}
_OUTPUT_FAILURE_EVENTS = {"attempt.output_publish.failed", "attempt.output_failed"}


@dataclass(frozen=True)
class SidecarSnapshot:
    name: str
    status: str = "unknown"
    sandbox_id: str | None = None
    provider: str | None = None


@dataclass(frozen=True)
class AttemptSnapshot:
    run_id: str
    agent_name: str
    task_id: str
    attempt_id: str
    status: str = "unknown"
    score: float | None = None
    sandbox_id: str | None = None
    gateway_session_id: str | None = None
    gateway_activity_available: bool | None = None
    gateway_request_count: int = 0
    gateway_last_request_time: str = ""
    gateway_response_status: int | None = None
    gateway_total_time_ms: int | None = None
    sidecars: list[SidecarSnapshot] = field(default_factory=list)
    quality_tags: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    events_seen: int = 0


def reduce_attempts(
    events: Iterable[EventEnvelope | Mapping[str, Any]],
) -> dict[str, AttemptSnapshot]:
    snapshots: dict[str, AttemptSnapshot] = {}
    for event in events:
        data = event.to_dict() if isinstance(event, EventEnvelope) else dict(event)
        name = str(data.get("name", ""))
        if _is_run_level_issue_event(name):
            continue
        context = dict(data.get("context") or {})
        attempt_id = context.get("attempt_id")
        task_id = context.get("task_id")
        run_id = context.get("run_id")
        if not attempt_id or not task_id or not run_id:
            continue

        key = _snapshot_key(
            snapshots,
            run_id=str(run_id),
            agent_name=str(context.get("agent_name") or ""),
            task_id=str(task_id),
            attempt_id=str(attempt_id),
        )
        previous = snapshots.get(
            key,
            AttemptSnapshot(
                run_id=str(run_id),
                agent_name=str(context.get("agent_name") or ""),
                task_id=str(task_id),
                attempt_id=str(attempt_id),
            ),
        )
        snapshots[key] = _reduce_one(previous, data)
    return snapshots


def _is_run_level_issue_event(name: str) -> bool:
    return name in {
        "process.event_forward_failed",
        "process.missing_result",
        "process.result_send_failed",
        "run_store.event_record_failed",
    }


def _snapshot_key(
    snapshots: dict[str, AttemptSnapshot],
    *,
    run_id: str,
    agent_name: str,
    task_id: str,
    attempt_id: str,
) -> str:
    previous = snapshots.get(attempt_id)
    if previous is None:
        if any(snapshot.attempt_id == attempt_id for snapshot in snapshots.values()):
            return _compound_attempt_key(run_id, agent_name, task_id, attempt_id)
        return attempt_id
    if previous.run_id == run_id and previous.agent_name == agent_name and previous.task_id == task_id:
        return attempt_id

    previous_key = _compound_attempt_key(
        previous.run_id,
        previous.agent_name,
        previous.task_id,
        previous.attempt_id,
    )
    snapshots[previous_key] = previous
    del snapshots[attempt_id]
    return _compound_attempt_key(run_id, agent_name, task_id, attempt_id)


def _compound_attempt_key(
    run_id: str,
    agent_name: str,
    task_id: str,
    attempt_id: str,
) -> str:
    return json.dumps(
        [run_id, agent_name, task_id, attempt_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _reduce_one(snapshot: AttemptSnapshot, event: dict[str, Any]) -> AttemptSnapshot:
    name = str(event.get("name", ""))
    context = dict(event.get("context") or {})
    payload = dict(event.get("payload") or {})
    status = snapshot.status
    score = snapshot.score
    sandbox_id = snapshot.sandbox_id or _optional_str(context.get("sandbox_id"))
    gateway_session_id = snapshot.gateway_session_id or _optional_str(context.get("gateway_session_id"))
    gateway_activity_available = snapshot.gateway_activity_available
    gateway_request_count = snapshot.gateway_request_count
    gateway_last_request_time = snapshot.gateway_last_request_time
    gateway_response_status = snapshot.gateway_response_status
    gateway_total_time_ms = snapshot.gateway_total_time_ms
    quality_tags = list(snapshot.quality_tags)
    errors = list(snapshot.errors)
    sidecars = {sidecar.name: sidecar for sidecar in snapshot.sidecars}

    if name in _STATUS_EVENTS:
        status = _STATUS_EVENTS[name]
    if name == "sandbox.created":
        sandbox_id = _optional_str(payload.get("sandbox_id")) or sandbox_id
    elif name in {"gateway.opened", "gateway.ready"}:
        gateway_session_id = _optional_str(payload.get("gateway_session_id")) or gateway_session_id
    elif name == "gateway.activity":
        gateway_activity_available = bool(payload.get("available", True))
        gateway_request_count = _optional_int(payload.get("request_count")) or 0
        gateway_last_request_time = str(payload.get("last_request_time") or "")
        gateway_response_status = _optional_int(payload.get("response_status"))
        gateway_total_time_ms = _optional_int(payload.get("total_time_ms"))
    elif name in _SIDECAR_EVENTS:
        sidecars = _record_sidecar(sidecars, payload, status=_SIDECAR_EVENTS[name])
    elif name == "sandbox.deleted" and payload.get("resource_role") == "sidecar":
        sidecars = _record_sidecar(
            sidecars,
            {
                "name": payload.get("resource_name"),
                "sandbox_id": payload.get("sandbox_id"),
                "provider": payload.get("provider"),
            },
            status="deleted",
        )
    elif name in _ARTIFACT_FAILURE_EVENTS:
        status = "artifact_pending"
        _append_payload_errors(errors, payload)
    elif name == "attempt.retrying":
        status = "retrying"
        errors.append(payload)
    elif name in {"attempt.failed", "attempt.cancelled"}:
        status = _status_from_error_payload(payload)
        errors.append(payload)
    elif name in _OUTPUT_FAILURE_EVENTS:
        status = "failed"
        errors.append(payload)
    elif name == "cleanup.failed":
        status = "cleanup_pending"
        _append_payload_errors(errors, payload)
        sidecars = _record_sidecar_cleanup(sidecars, payload)
    elif name == "gateway.trace_export.failed":
        errors.append(payload)
    elif name == "attempt.evaluated":
        status = "evaluated"
        score = _optional_float(payload.get("score"))
    elif name == "attempt.completed":
        status = str(payload.get("status") or "completed")
        score = _optional_float(payload.get("score"))
        quality_tags = [str(tag) for tag in payload.get("quality_tags", [])]
    elif name == "attempt.output_published":
        status = str(payload.get("status") or status)
        score = _optional_float(payload.get("score")) if payload.get("score") is not None else score

    return AttemptSnapshot(
        run_id=snapshot.run_id,
        agent_name=snapshot.agent_name,
        task_id=snapshot.task_id,
        attempt_id=snapshot.attempt_id,
        status=status,
        score=score,
        sandbox_id=sandbox_id,
        gateway_session_id=gateway_session_id,
        gateway_activity_available=gateway_activity_available,
        gateway_request_count=gateway_request_count,
        gateway_last_request_time=gateway_last_request_time,
        gateway_response_status=gateway_response_status,
        gateway_total_time_ms=gateway_total_time_ms,
        sidecars=list(sidecars.values()),
        quality_tags=quality_tags,
        errors=errors,
        events_seen=snapshot.events_seen + 1,
    )


def _record_sidecar(
    sidecars: dict[str, SidecarSnapshot],
    payload: dict[str, Any],
    *,
    status: str,
) -> dict[str, SidecarSnapshot]:
    name = _optional_str(payload.get("name"))
    if not name:
        return sidecars
    previous = sidecars.get(name, SidecarSnapshot(name=name))
    sidecars[name] = SidecarSnapshot(
        name=name,
        status=status,
        sandbox_id=_optional_str(payload.get("sandbox_id")) or previous.sandbox_id,
        provider=_optional_str(payload.get("provider")) or previous.provider,
    )
    return sidecars


def _record_sidecar_cleanup(
    sidecars: dict[str, SidecarSnapshot],
    payload: dict[str, Any],
) -> dict[str, SidecarSnapshot]:
    error_payloads = [payload]
    nested_errors = payload.get("errors")
    if isinstance(nested_errors, list):
        error_payloads.extend(error for error in nested_errors if isinstance(error, Mapping))
    for error in error_payloads:
        details = dict(error.get("details") or {})
        if details.get("resource_role") != "sidecar":
            continue
        name = _optional_str(details.get("resource_name"))
        if not name:
            continue
        previous = sidecars.get(name, SidecarSnapshot(name=name))
        sidecars[name] = SidecarSnapshot(
            name=name,
            status="cleanup_pending",
            sandbox_id=(_optional_str(details.get("sandbox_id") or details.get("resource_id")) or previous.sandbox_id),
            provider=_optional_str(details.get("provider")) or previous.provider,
        )
    return sidecars


def _append_payload_errors(errors: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    nested = payload.get("errors")
    if isinstance(nested, list):
        errors.extend(error for error in nested if isinstance(error, dict))
    else:
        errors.append(payload)


def _status_from_error_payload(payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "")
    if status in {"cancelled", "timeout", "failed"}:
        return status
    if payload.get("kind") == "cancelled":
        return "cancelled"
    if payload.get("kind") == "timeout":
        return "timeout"
    return "failed"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
