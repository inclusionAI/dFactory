"""Pure TUI view models built from event-derived live state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from anexus.events import EventEnvelope
from anexus.ui.live import RunEventBuffer, RunLiveState, build_run_live_state


@dataclass(frozen=True)
class TUIAttemptRow:
    """Compact attempt row for terminal renderers."""

    agent_name: str
    task_id: str
    attempt_id: str
    status: str
    score: float | None = None
    sandbox_id: str = ""
    gateway_session_id: str = ""
    gateway_activity_available: bool | None = None
    gateway_request_count: int = 0
    gateway_last_request_time: str = ""
    gateway_response_status: int | None = None
    gateway_total_time_ms: int | None = None
    error_count: int = 0
    quality_tags: list[str] = field(default_factory=list)
    error_kinds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TUIGatewayRow:
    """Compact gateway session row for terminal renderers."""

    gateway_session_id: str
    status: str
    phase: str = ""
    upstream_proxy_events: int = 0
    upstream_proxy_errors: int = 0
    upstream_proxy_error_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TUISandboxResourceRow:
    """Compact sandbox resource row for terminal renderers."""

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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TUIRunIssueRow:
    """Compact run issue row for terminal renderers."""

    reason: str
    event_name: str
    count: int = 1
    agent_name: str = ""
    task_id: str = ""
    attempt_id: str = ""
    phase: str = ""
    message: str = ""
    error_type: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TUIRunView:
    """Serializable run view consumed by TUI renderers."""

    run_id: str
    agent_names: list[str]
    events_seen: int
    status_counts: dict[str, int]
    total_tokens: int
    attempts_queued: int
    attempts_admitted: int
    queue_wait_seconds_total: float
    global_wait_seconds_total: float
    agent_wait_seconds_total: float
    cleanup_pending: int
    artifact_pending: int
    error_kind_counts: dict[str, int]
    run_issue_counts: dict[str, int]
    gateways: list[TUIGatewayRow]
    sandbox_resources: list[TUISandboxResourceRow]
    run_issues: list[TUIRunIssueRow]
    attempts: list[TUIAttemptRow]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "agent_names": list(self.agent_names),
            "events_seen": self.events_seen,
            "status_counts": dict(self.status_counts),
            "total_tokens": self.total_tokens,
            "attempts_queued": self.attempts_queued,
            "attempts_admitted": self.attempts_admitted,
            "queue_wait_seconds_total": self.queue_wait_seconds_total,
            "global_wait_seconds_total": self.global_wait_seconds_total,
            "agent_wait_seconds_total": self.agent_wait_seconds_total,
            "cleanup_pending": self.cleanup_pending,
            "artifact_pending": self.artifact_pending,
            "error_kind_counts": dict(self.error_kind_counts),
            "run_issue_counts": dict(self.run_issue_counts),
            "gateways": [gateway.to_dict() for gateway in self.gateways],
            "sandbox_resources": [resource.to_dict() for resource in self.sandbox_resources],
            "run_issues": [issue.to_dict() for issue in self.run_issues],
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


class TUIEventView:
    """Event-bus sink that exposes a TUI view without importing runtime code."""

    def __init__(self) -> None:
        self._buffer = RunEventBuffer()

    def emit(self, event: EventEnvelope) -> None:
        self._buffer.emit(event)

    def append(self, event: EventEnvelope | Mapping[str, Any]) -> None:
        self._buffer.append(event)

    def extend(self, events: Iterable[EventEnvelope | Mapping[str, Any]]) -> None:
        self._buffer.extend(events)

    def snapshot(self) -> TUIRunView:
        return build_tui_view(self._buffer.snapshot())


def build_tui_view(state: RunLiveState) -> TUIRunView:
    """Build a terminal-oriented view model from live run state."""

    return TUIRunView(
        run_id=state.run_id,
        agent_names=list(state.agent_names),
        events_seen=state.events_seen,
        status_counts=dict(state.status_counts),
        total_tokens=state.total_tokens,
        attempts_queued=state.attempts_queued,
        attempts_admitted=state.attempts_admitted,
        queue_wait_seconds_total=state.queue_wait_seconds_total,
        global_wait_seconds_total=state.global_wait_seconds_total,
        agent_wait_seconds_total=state.agent_wait_seconds_total,
        cleanup_pending=state.cleanup_pending,
        artifact_pending=state.artifact_pending,
        error_kind_counts=dict(state.error_kind_counts),
        run_issue_counts=dict(state.run_issue_counts),
        gateways=[
            TUIGatewayRow(
                gateway_session_id=gateway.gateway_session_id,
                status=gateway.status,
                phase=gateway.phase,
                upstream_proxy_events=gateway.upstream_proxy_events,
                upstream_proxy_errors=gateway.upstream_proxy_errors,
                upstream_proxy_error_counts=dict(gateway.upstream_proxy_error_counts),
            )
            for gateway in state.gateways
        ],
        sandbox_resources=[
            TUISandboxResourceRow(
                provider_name=resource.provider_name,
                phase=resource.phase,
                status=resource.status,
                node_count=resource.node_count,
                ready_node_count=resource.ready_node_count,
                cpu_capacity_m=resource.cpu_capacity_m,
                cpu_allocatable_m=resource.cpu_allocatable_m,
                cpu_used_m=resource.cpu_used_m,
                memory_capacity_mb=resource.memory_capacity_mb,
                memory_allocatable_mb=resource.memory_allocatable_mb,
                memory_used_mb=resource.memory_used_mb,
                reason=resource.reason,
                error_type=resource.error_type,
                error=resource.error,
            )
            for resource in state.sandbox_resources
        ],
        run_issues=[
            TUIRunIssueRow(
                reason=issue.reason,
                event_name=issue.event_name,
                count=issue.count,
                agent_name=issue.agent_name,
                task_id=issue.task_id,
                attempt_id=issue.attempt_id,
                phase=issue.phase,
                message=issue.message,
                error_type=issue.error_type,
            )
            for issue in state.run_issues
        ],
        attempts=[
            TUIAttemptRow(
                agent_name=attempt.agent_name,
                task_id=attempt.task_id,
                attempt_id=attempt.attempt_id,
                status=attempt.status,
                score=attempt.score,
                sandbox_id=attempt.sandbox_id or "",
                gateway_session_id=attempt.gateway_session_id or "",
                gateway_activity_available=attempt.gateway_activity_available,
                gateway_request_count=attempt.gateway_request_count,
                gateway_last_request_time=attempt.gateway_last_request_time,
                gateway_response_status=attempt.gateway_response_status,
                gateway_total_time_ms=attempt.gateway_total_time_ms,
                error_count=attempt.error_count,
                quality_tags=list(attempt.quality_tags),
                error_kinds=list(attempt.error_kinds),
            )
            for attempt in state.attempts
        ],
    )


def build_tui_view_from_events(
    events: Iterable[EventEnvelope | Mapping[str, Any]],
) -> TUIRunView:
    """Build a TUI view directly from event envelopes or replayed event rows."""

    return build_tui_view(build_run_live_state(events))


def render_tui_text(view: TUIRunView) -> str:
    """Render a stable plain-text TUI snapshot."""

    status_text = ", ".join(f"{name}={count}" for name, count in sorted(view.status_counts.items()))
    if not status_text:
        status_text = "none"
    error_text = ", ".join(f"{name}={count}" for name, count in sorted(view.error_kind_counts.items()))
    if not error_text:
        error_text = "none"
    issue_text = ", ".join(f"{name}={count}" for name, count in sorted(view.run_issue_counts.items()))
    if not issue_text:
        issue_text = "none"
    lines = [
        (
            f"run={view.run_id or '-'} agents={_cell(','.join(view.agent_names))} "
            f"events={view.events_seen} "
            f"tokens={view.total_tokens} status={status_text} "
            f"queued={view.attempts_queued} admitted={view.attempts_admitted} "
            f"queue_wait={_seconds_text(view.queue_wait_seconds_total)} "
            f"global_wait={_seconds_text(view.global_wait_seconds_total)} "
            f"agent_wait={_seconds_text(view.agent_wait_seconds_total)} "
            f"cleanup_pending={view.cleanup_pending} artifact_pending={view.artifact_pending} "
            f"errors={error_text} issues={issue_text}"
        ),
        "agent_name task_id attempt_id status score sandbox gateway requests llm_last latest errors tags",
    ]
    for gateway in view.gateways:
        lines.append(_gateway_row_text(gateway))
    for resource in view.sandbox_resources:
        lines.append(_sandbox_resource_row_text(resource))
    for issue in view.run_issues:
        lines.append(_run_issue_row_text(issue))
    for row in view.attempts:
        lines.append(
            " ".join(
                [
                    _cell(row.agent_name),
                    _cell(row.task_id),
                    _cell(row.attempt_id),
                    _cell(row.status),
                    _score_text(row.score),
                    _cell(row.sandbox_id),
                    _cell(row.gateway_session_id),
                    str(row.gateway_request_count),
                    _cell(row.gateway_last_request_time),
                    _gateway_latest_text(row),
                    str(row.error_count),
                    _cell(",".join(row.quality_tags + row.error_kinds)),
                ]
            )
        )
    return "\n".join(lines)


def _run_issue_row_text(row: TUIRunIssueRow) -> str:
    parts = [
        "issue",
        _cell(row.reason),
        f"count={row.count}",
        f"event={_cell(row.event_name)}",
        f"agent={_cell(row.agent_name)}",
        f"task={_cell(row.task_id)}",
        f"attempt={_cell(row.attempt_id)}",
        f"phase={_cell(row.phase)}",
    ]
    if row.error_type:
        parts.append(f"error_type={row.error_type}")
    if row.message:
        parts.append(f"message={row.message}")
    return " ".join(parts)


def _sandbox_resource_row_text(row: TUISandboxResourceRow) -> str:
    parts = [
        "sandbox_resources",
        f"provider={_cell(row.provider_name)}",
        f"status={_cell(row.status)}",
    ]
    if row.phase:
        parts.append(f"phase={row.phase}")
    if row.status == "unavailable":
        if row.reason:
            parts.append(f"reason={row.reason}")
        if row.error_type:
            parts.append(f"error_type={row.error_type}")
        if row.error:
            parts.append(f"error={row.error}")
        return " ".join(parts)

    parts.extend(
        [
            f"nodes={row.ready_node_count}/{row.node_count}",
            (f"cpu={_resource_amount(row.cpu_used_m)}/{_resource_amount(row.cpu_capacity_m)}m"),
            (f"memory={_resource_amount(row.memory_used_mb)}/{_resource_amount(row.memory_capacity_mb)}MB"),
        ]
    )
    if row.reason:
        parts.append(f"reason={row.reason}")
    return " ".join(parts)


def _gateway_row_text(row: TUIGatewayRow) -> str:
    counts = ", ".join(f"{reason}:{count}" for reason, count in sorted(row.upstream_proxy_error_counts.items()))
    counts = f" ({counts})" if counts else ""
    phase = f" phase={row.phase}" if row.phase else ""
    return (
        f"gateway {row.gateway_session_id} status={row.status}{phase} "
        f"upstream_proxy_events={row.upstream_proxy_events} "
        f"upstream_proxy_errors={row.upstream_proxy_errors}{counts}"
    )


def _score_text(score: float | None) -> str:
    if score is None:
        return "-"
    return f"{score:.3f}".rstrip("0").rstrip(".")


def _gateway_latest_text(row: TUIAttemptRow) -> str:
    if row.gateway_activity_available is False:
        return "unknown"
    status = str(row.gateway_response_status) if row.gateway_response_status is not None else "-"
    latency = f"{row.gateway_total_time_ms}ms" if row.gateway_total_time_ms is not None else "-"
    return f"{status}/{latency}"


def _resource_amount(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


def _seconds_text(value: float) -> str:
    numeric = float(value)
    if numeric == 0:
        return "0s"
    return f"{numeric:.3f}".rstrip("0").rstrip(".") + "s"


def _cell(value: str) -> str:
    return value if value else "-"
