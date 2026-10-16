"""Live run state rebuilt only from events."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from anexus.events import AttemptSnapshot, EventEnvelope, reduce_attempts
from anexus.metrics import RunMetrics, reduce_run_metrics


@dataclass(frozen=True)
class SidecarLiveState:
    name: str
    status: str
    sandbox_id: str | None = None
    provider: str | None = None


@dataclass(frozen=True)
class AttemptLiveState:
    run_id: str
    agent_name: str
    task_id: str
    attempt_id: str
    status: str
    score: float | None = None
    sandbox_id: str | None = None
    gateway_session_id: str | None = None
    gateway_activity_available: bool | None = None
    gateway_request_count: int = 0
    gateway_last_request_time: str = ""
    gateway_response_status: int | None = None
    gateway_total_time_ms: int | None = None
    sidecars: list[SidecarLiveState] = field(default_factory=list)
    quality_tags: list[str] = field(default_factory=list)
    error_count: int = 0
    error_kinds: list[str] = field(default_factory=list)
    events_seen: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GatewayLiveState:
    gateway_session_id: str
    status: str = "unknown"
    phase: str = ""
    upstream_proxy_events: int = 0
    upstream_proxy_errors: int = 0
    upstream_proxy_error_counts: dict[str, int] = field(default_factory=dict)
    recent_upstream_proxy_errors: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SandboxResourceLiveState:
    provider_name: str
    phase: str = ""
    status: str = "ready"
    node_count: int = 0
    ready_node_count: int = 0
    cpu_capacity_m: float = 0.0
    cpu_allocatable_m: float = 0.0
    cpu_used_m: float = 0.0
    memory_capacity_mb: float = 0.0
    memory_allocatable_mb: float = 0.0
    memory_used_mb: float = 0.0
    reason: str = ""
    error_type: str = ""
    error: str = ""
    details: dict[str, object] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RunIssueLiveState:
    reason: str
    event_name: str
    count: int = 1
    run_id: str = ""
    agent_name: str = ""
    task_id: str = ""
    attempt_id: str = ""
    phase: str = ""
    message: str = ""
    error_type: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RunLiveState:
    run_id: str
    agent_names: list[str]
    attempts: list[AttemptLiveState]
    gateways: list[GatewayLiveState]
    sandbox_resources: list[SandboxResourceLiveState]
    run_issues: list[RunIssueLiveState]
    status_counts: dict[str, int]
    run_issue_counts: dict[str, int]
    events_seen: int
    attempts_queued: int
    attempts_admitted: int
    attempts_started: int
    attempts_retrying: int
    attempts_failed: int
    cleanup_pending: int
    artifact_pending: int
    error_kind_counts: dict[str, int]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    queue_wait_seconds_total: float
    global_wait_seconds_total: float
    agent_wait_seconds_total: float

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "agent_names": list(self.agent_names),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "gateways": [gateway.to_dict() for gateway in self.gateways],
            "sandbox_resources": [resource.to_dict() for resource in self.sandbox_resources],
            "run_issues": [issue.to_dict() for issue in self.run_issues],
            "status_counts": dict(self.status_counts),
            "run_issue_counts": dict(self.run_issue_counts),
            "events_seen": self.events_seen,
            "attempts_queued": self.attempts_queued,
            "attempts_admitted": self.attempts_admitted,
            "attempts_started": self.attempts_started,
            "attempts_retrying": self.attempts_retrying,
            "attempts_failed": self.attempts_failed,
            "cleanup_pending": self.cleanup_pending,
            "artifact_pending": self.artifact_pending,
            "error_kind_counts": dict(self.error_kind_counts),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "queue_wait_seconds_total": self.queue_wait_seconds_total,
            "global_wait_seconds_total": self.global_wait_seconds_total,
            "agent_wait_seconds_total": self.agent_wait_seconds_total,
        }


class RunEventBuffer:
    """Append-only event buffer for live UI consumers."""

    def __init__(self) -> None:
        self._events: list[EventEnvelope | Mapping[str, Any]] = []

    def append(self, event: EventEnvelope | Mapping[str, Any]) -> None:
        self._events.append(event)

    def emit(self, event: EventEnvelope) -> None:
        self.append(event)

    def extend(self, events: Iterable[EventEnvelope | Mapping[str, Any]]) -> None:
        self._events.extend(events)

    def snapshot(self) -> RunLiveState:
        return build_run_live_state(self._events)

    def clear(self) -> None:
        self._events.clear()


def build_run_live_state(events: Iterable[EventEnvelope | Mapping[str, Any]]) -> RunLiveState:
    event_list = list(events)
    metrics = reduce_run_metrics(event_list)
    attempts = [
        _attempt_state(snapshot)
        for snapshot in sorted(
            reduce_attempts(event_list).values(),
            key=lambda item: (item.agent_name, item.task_id, item.attempt_id),
        )
    ]
    gateways = _gateway_states(event_list)
    sandbox_resources = _sandbox_resource_states(event_list)
    run_issues = _run_issue_states(event_list)
    return _run_state(
        metrics=metrics,
        agent_names=_agent_names(event_list, attempts),
        attempts=attempts,
        gateways=gateways,
        sandbox_resources=sandbox_resources,
        run_issues=run_issues,
    )


def _attempt_state(snapshot: AttemptSnapshot) -> AttemptLiveState:
    return AttemptLiveState(
        run_id=snapshot.run_id,
        agent_name=snapshot.agent_name,
        task_id=snapshot.task_id,
        attempt_id=snapshot.attempt_id,
        status=snapshot.status,
        score=snapshot.score,
        sandbox_id=snapshot.sandbox_id,
        gateway_session_id=snapshot.gateway_session_id,
        gateway_activity_available=snapshot.gateway_activity_available,
        gateway_request_count=snapshot.gateway_request_count,
        gateway_last_request_time=snapshot.gateway_last_request_time,
        gateway_response_status=snapshot.gateway_response_status,
        gateway_total_time_ms=snapshot.gateway_total_time_ms,
        sidecars=[
            SidecarLiveState(
                name=sidecar.name,
                status=sidecar.status,
                sandbox_id=sidecar.sandbox_id,
                provider=sidecar.provider,
            )
            for sidecar in snapshot.sidecars
        ],
        quality_tags=list(snapshot.quality_tags),
        error_count=len(snapshot.errors),
        error_kinds=_error_kinds(snapshot.errors),
        events_seen=snapshot.events_seen,
    )


def _run_state(
    *,
    metrics: RunMetrics,
    agent_names: list[str],
    attempts: list[AttemptLiveState],
    gateways: list[GatewayLiveState],
    sandbox_resources: list[SandboxResourceLiveState],
    run_issues: list[RunIssueLiveState],
) -> RunLiveState:
    run_id = metrics.run_id or (attempts[0].run_id if attempts else "")
    run_issue_counts = Counter[str]()
    for issue in run_issues:
        run_issue_counts[issue.reason] += issue.count
    return RunLiveState(
        run_id=run_id,
        agent_names=agent_names,
        attempts=attempts,
        gateways=gateways,
        sandbox_resources=sandbox_resources,
        run_issues=run_issues,
        status_counts=dict(metrics.status_counts),
        run_issue_counts=dict(sorted(run_issue_counts.items())),
        events_seen=metrics.events_seen,
        attempts_queued=metrics.attempts_queued,
        attempts_admitted=metrics.attempts_admitted,
        attempts_started=metrics.attempts_started,
        attempts_retrying=metrics.attempts_retrying,
        attempts_failed=metrics.attempts_failed,
        cleanup_pending=metrics.cleanup_pending,
        artifact_pending=metrics.artifact_pending,
        error_kind_counts=dict(metrics.error_kind_counts),
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        total_tokens=metrics.total_tokens,
        queue_wait_seconds_total=metrics.queue_wait_seconds_total,
        global_wait_seconds_total=metrics.global_wait_seconds_total,
        agent_wait_seconds_total=metrics.agent_wait_seconds_total,
    )


def _agent_names(
    events: list[EventEnvelope | Mapping[str, Any]],
    attempts: list[AttemptLiveState],
) -> list[str]:
    names: list[str] = []
    for event in events:
        data = event.to_dict() if isinstance(event, EventEnvelope) else dict(event)
        payload = dict(data.get("payload") or {})
        raw_names = payload.get("agent_names") if data.get("name") == "run.started" else None
        if isinstance(raw_names, list):
            for raw_name in raw_names:
                name = str(raw_name).strip()
                if name and name not in names:
                    names.append(name)
    for attempt in attempts:
        if attempt.agent_name and attempt.agent_name not in names:
            names.append(attempt.agent_name)
    return names


def _error_kinds(errors: list[dict[str, Any]]) -> list[str]:
    kinds: list[str] = []
    for error in errors:
        kind = error.get("kind")
        if kind is None:
            continue
        kind_text = str(kind)
        if kind_text not in kinds:
            kinds.append(kind_text)
    return kinds


def _gateway_states(events: list[EventEnvelope | Mapping[str, Any]]) -> list[GatewayLiveState]:
    states: dict[str, GatewayLiveState] = {}
    for event in events:
        data = event.to_dict() if isinstance(event, EventEnvelope) else dict(event)
        context = dict(data.get("context") or {})
        payload = dict(data.get("payload") or {})
        session_id = _gateway_session_id(context, payload)
        if not session_id:
            continue
        previous = states.get(session_id, GatewayLiveState(gateway_session_id=session_id))
        states[session_id] = _reduce_gateway_state(previous, data)
    return [
        states[session_id]
        for session_id in sorted(states)
        if states[session_id].upstream_proxy_events or states[session_id].status != "unknown"
    ]


def _sandbox_resource_states(
    events: list[EventEnvelope | Mapping[str, Any]],
) -> list[SandboxResourceLiveState]:
    states: dict[str, SandboxResourceLiveState] = {}
    for event in events:
        data = event.to_dict() if isinstance(event, EventEnvelope) else dict(event)
        if str(data.get("name", "")) != "sandbox.resources.snapshot":
            continue
        payload = dict(data.get("payload") or {})
        details = dict(payload.get("details") or {})
        provider_name = _optional_resource_text(payload.get("provider_name") or payload.get("provider"))
        if not provider_name:
            provider_name = "sandbox"
        states[provider_name] = SandboxResourceLiveState(
            provider_name=provider_name,
            phase=_optional_resource_text(payload.get("phase")),
            status=_optional_resource_text(payload.get("status") or details.get("status")) or "ready",
            node_count=_int_value(payload.get("node_count")),
            ready_node_count=_int_value(payload.get("ready_node_count")),
            cpu_capacity_m=_float_value(payload.get("cpu_capacity_m")),
            cpu_allocatable_m=_float_value(payload.get("cpu_allocatable_m")),
            cpu_used_m=_float_value(payload.get("cpu_used_m")),
            memory_capacity_mb=_float_value(payload.get("memory_capacity_mb")),
            memory_allocatable_mb=_float_value(payload.get("memory_allocatable_mb")),
            memory_used_mb=_float_value(payload.get("memory_used_mb")),
            reason=_optional_resource_text(payload.get("reason") or details.get("reason")),
            error_type=_optional_resource_text(payload.get("error_type") or details.get("error_type")),
            error=_optional_resource_text(payload.get("error") or details.get("error")),
            details=details,
            timestamp=_optional_resource_text(data.get("timestamp")),
        )
    return [states[provider_name] for provider_name in sorted(states)]


def _run_issue_states(events: list[EventEnvelope | Mapping[str, Any]]) -> list[RunIssueLiveState]:
    states: dict[tuple[str, str, str, str, str, str, str, str, str], RunIssueLiveState] = {}
    for event in events:
        data = event.to_dict() if isinstance(event, EventEnvelope) else dict(event)
        name = str(data.get("name", ""))
        reason = _run_issue_reason(name)
        if not reason:
            continue
        context = dict(data.get("context") or {})
        payload = dict(data.get("payload") or {})
        details = dict(payload.get("details") or {})
        original_error = dict(details.get("original_error") or {})
        issue = RunIssueLiveState(
            reason=reason,
            event_name=name,
            run_id=_issue_value(context, payload, "run_id"),
            agent_name=_issue_value(context, payload, "agent_name"),
            task_id=_issue_value(context, payload, "task_id"),
            attempt_id=_issue_value(context, payload, "attempt_id"),
            phase=_issue_phase(payload, details),
            message=_issue_message(payload, details, original_error),
            error_type=_issue_error_type(payload, details, original_error),
        )
        key = (
            issue.reason,
            issue.event_name,
            issue.run_id,
            issue.agent_name,
            issue.task_id,
            issue.attempt_id,
            issue.phase,
            issue.message,
            issue.error_type,
        )
        previous = states.get(key)
        if previous is None:
            states[key] = issue
        else:
            states[key] = RunIssueLiveState(
                reason=previous.reason,
                event_name=previous.event_name,
                count=previous.count + 1,
                run_id=previous.run_id,
                agent_name=previous.agent_name,
                task_id=previous.task_id,
                attempt_id=previous.attempt_id,
                phase=previous.phase,
                message=previous.message,
                error_type=previous.error_type,
            )
    return [
        states[key]
        for key in sorted(
            states,
            key=lambda item: (item[0], item[2], item[3], item[4], item[5], item[1], item[6]),
        )
    ]


def _run_issue_reason(name: str) -> str:
    return {
        "process.event_forward_failed": "event_forward_failed",
        "process.missing_result": "process_missing_result",
        "process.result_send_failed": "process_result_send_failed",
        "run_store.event_record_failed": "run_store_event_record_failed",
    }.get(name, "")


def _issue_value(
    context: Mapping[str, Any],
    payload: Mapping[str, Any],
    field: str,
) -> str:
    return str(context.get(field) or payload.get(field) or "")


def _issue_phase(payload: Mapping[str, Any], details: Mapping[str, Any]) -> str:
    return str(payload.get("phase") or payload.get("message_kind") or details.get("failed_event_name") or "")


def _issue_message(
    payload: Mapping[str, Any],
    details: Mapping[str, Any],
    original_error: Mapping[str, Any],
) -> str:
    return str(payload.get("message") or original_error.get("message") or details.get("message") or "")


def _issue_error_type(
    payload: Mapping[str, Any],
    details: Mapping[str, Any],
    original_error: Mapping[str, Any],
) -> str:
    return str(
        payload.get("exception_type")
        or payload.get("error_type")
        or original_error.get("exception_type")
        or original_error.get("error_type")
        or details.get("exception_type")
        or details.get("error_type")
        or ""
    )


def _reduce_gateway_state(
    state: GatewayLiveState,
    event: dict[str, Any],
) -> GatewayLiveState:
    name = str(event.get("name", ""))
    payload = dict(event.get("payload") or {})
    status = state.status
    phase = state.phase
    events_seen = state.upstream_proxy_events
    error_count = state.upstream_proxy_errors
    error_counts = dict(state.upstream_proxy_error_counts)
    recent_errors = list(state.recent_upstream_proxy_errors)

    if name == "gateway.opened":
        status = "open"
    elif name in {"gateway.closed", "gateway.session.closed"}:
        status = "closed"
    elif name in {"gateway.failed", "gateway.session.error", "gateway.upstream_proxy.error"}:
        status = "error"
    elif name == "gateway.upstream_proxy.event" and status == "unknown":
        status = "active"

    summary = _upstream_proxy_summary(payload)
    if summary:
        events_seen += summary["total"]
        error_count += summary["error_count"]
        for reason, count in summary["error_counts"].items():
            error_counts[reason] = error_counts.get(reason, 0) + count
        recent_errors = [*recent_errors, *summary["recent_errors"]][-5:]
        if summary["error_count"] and status not in {"error", "closed"}:
            status = "warning"

    event_phase = str(payload.get("phase") or "")
    if event_phase:
        phase = event_phase
    elif name.startswith("gateway."):
        phase = name

    return GatewayLiveState(
        gateway_session_id=state.gateway_session_id,
        status=status,
        phase=phase,
        upstream_proxy_events=events_seen,
        upstream_proxy_errors=error_count,
        upstream_proxy_error_counts=dict(sorted(error_counts.items())),
        recent_upstream_proxy_errors=recent_errors,
    )


def _gateway_session_id(context: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    return str(
        context.get("gateway_session_id") or payload.get("gateway_session_id") or payload.get("session_id") or ""
    )


def _optional_resource_text(value: object) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _int_value(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float_value(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _upstream_proxy_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_summary = payload.get("upstream_proxy_events")
    if isinstance(raw_summary, Mapping):
        return _normalized_upstream_proxy_summary(raw_summary)
    raw_events = payload.get("events")
    if isinstance(raw_events, list):
        return _summary_from_proxy_events([dict(item) for item in raw_events if isinstance(item, Mapping)])
    if _is_upstream_proxy_payload(payload):
        return _summary_from_proxy_events([dict(payload)])
    return {}


def _normalized_upstream_proxy_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    error_counts = {str(reason): int(count) for reason, count in dict(summary.get("error_counts") or {}).items()}
    recent = summary.get("recent_errors")
    return {
        "total": int(summary.get("total") or 0),
        "error_count": int(summary.get("error_count") or sum(error_counts.values())),
        "error_counts": error_counts,
        "recent_errors": [dict(item) for item in recent if isinstance(item, Mapping)]
        if isinstance(recent, list)
        else [],
    }


def _summary_from_proxy_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    error_counts: dict[str, int] = {}
    recent_errors: list[dict[str, object]] = []
    for event in events:
        if not _is_upstream_proxy_error_event(event):
            continue
        bucket = _upstream_proxy_error_bucket(event)
        error_counts[bucket] = error_counts.get(bucket, 0) + 1
        recent_errors.append(_compact_upstream_proxy_error(event))
    return {
        "total": len(events),
        "error_count": sum(error_counts.values()),
        "error_counts": dict(sorted(error_counts.items())),
        "recent_errors": recent_errors[-5:],
    }


def _is_upstream_proxy_payload(payload: Mapping[str, Any]) -> bool:
    kind = str(payload.get("kind") or "")
    return kind.startswith("upstream") or any(
        payload.get(field)
        for field in (
            "failure_reason",
            "client_disconnected",
            "response_error",
            "upstream_close_error",
            "error_type",
            "error",
        )
    )


def _is_upstream_proxy_error_event(event: Mapping[str, Any]) -> bool:
    kind = str(event.get("kind") or "")
    if kind in {
        "malformed_request",
        "request_body_error",
        "route_not_found",
        "stream_error",
        "upstream_error",
    }:
        return True
    return any(
        event.get(field)
        for field in (
            "failure_reason",
            "client_disconnected",
            "response_error",
            "upstream_close_error",
            "error_type",
            "error",
        )
    )


def _upstream_proxy_error_bucket(event: Mapping[str, Any]) -> str:
    failure_reason = str(event.get("failure_reason") or "")
    if failure_reason:
        return failure_reason
    kind = str(event.get("kind") or "")
    if kind:
        return kind
    if event.get("client_disconnected"):
        return "client_disconnected"
    if event.get("response_error"):
        return "response_error"
    if event.get("upstream_close_error"):
        return "upstream_close_error"
    if event.get("error_type") or event.get("error"):
        return "error"
    return "unknown"


def _compact_upstream_proxy_error(event: Mapping[str, Any]) -> dict[str, object]:
    compact: dict[str, object] = {}
    for key in (
        "kind",
        "request_id",
        "route_key",
        "route",
        "status",
        "error_type",
        "error",
        "failure_reason",
        "retryable",
        "client_disconnected",
        "response_error",
        "upstream_close_error",
    ):
        value = event.get(key)
        if key == "retryable":
            if isinstance(value, bool):
                compact[key] = value
            continue
        if value not in (None, "", False):
            compact[key] = value
    return compact
