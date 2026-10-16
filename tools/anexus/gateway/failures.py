"""Gateway failure classification helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

TASK_AFFECTING_GATEWAY_REASONS = frozenset(
    {
        "llm_rate_limited",
        "llm_upstream_unavailable",
        "llm_upstream_failed",
        "llm_http_error",
        "llm_stream_failed",
        "llm_invalid_json",
        "llm_malformed_response",
        "mirror_route_missing",
        "mirror_protocol_unsupported",
    }
)

TASK_AFFECTING_GATEWAY_PHASES = frozenset(
    {
        "gateway.engine.forward",
        "gateway.mirror.forward",
        "gateway.mirror.route",
        "gateway.mirror.stream",
        "gateway.mirror.stream.connect",
        "gateway.kgb.create",
        "gateway.kgb.checkout",
        "gateway.kgb.trace_export",
        "gateway.kgb.retire",
    }
)


@dataclass(frozen=True)
class GatewayFailure:
    """Describe a normalized gateway request failure."""

    phase: str
    reason: str
    retryable: bool
    status_code: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable failure description."""
        return {
            "phase": self.phase,
            "reason": self.reason,
            "retryable": self.retryable,
            "status_code": self.status_code,
        }


@dataclass(frozen=True)
class UpstreamProxyFailure:
    """Normalized failure classification for upstream proxy events."""

    reason: str
    retryable: bool


_RETRYABLE_UPSTREAM_ERROR_TYPES = frozenset(
    {
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "OSError",
        "RemoteDisconnected",
        "TimeoutError",
        "URLError",
    }
)

_RATE_LIMIT_PATTERNS = (
    "429",
    "rate limit",
    "ratelimit",
    "too many requests",
    "quota exceeded",
    "throttle",
    "throttled",
)

_UNAVAILABLE_PATTERNS = (
    "connection refused",
    "connection reset",
    "temporarily unavailable",
    "service unavailable",
    "upstream unavailable",
    "bad gateway",
    "gateway timeout",
    "timed out",
    "timeout",
)


def classify_gateway_http_failure(
    *,
    phase: str,
    status_code: int,
) -> GatewayFailure:
    """Classify HTTP failures from engine or mirror gateway calls."""
    if status_code == 429:
        return GatewayFailure(
            phase=phase,
            reason="llm_rate_limited",
            retryable=True,
            status_code=status_code,
        )
    if 500 <= status_code <= 599:
        return GatewayFailure(
            phase=phase,
            reason="llm_upstream_unavailable",
            retryable=True,
            status_code=status_code,
        )
    return GatewayFailure(
        phase=phase,
        reason="llm_http_error",
        retryable=False,
        status_code=status_code,
    )


def classify_gateway_transport_failure(
    *,
    phase: str,
    exception_type: str,
) -> GatewayFailure:
    """Classify connection-level failures from gateway forwarding."""
    retryable = exception_type in {
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "OSError",
        "RemoteDisconnected",
        "TimeoutError",
        "URLError",
    }
    return GatewayFailure(
        phase=phase,
        reason="llm_upstream_unavailable" if retryable else "llm_upstream_failed",
        retryable=retryable,
        status_code=502,
    )


def classify_upstream_proxy_failure(
    *,
    kind: str,
    error_type: str = "",
    error: str = "",
    status: object = None,
    client_disconnected: bool = False,
    response_error: str = "",
    upstream_close_error: str = "",
) -> UpstreamProxyFailure | None:
    """Classify host-side proxy events into stable gateway failure reasons."""
    if client_disconnected:
        return UpstreamProxyFailure("client_disconnected", False)
    if kind == "route_not_found":
        return UpstreamProxyFailure("mirror_route_missing", False)
    if kind == "request_body_error":
        return UpstreamProxyFailure("request_body_failed", False)
    if kind == "malformed_request":
        return UpstreamProxyFailure("malformed_request", False)

    status_value = _coerce_status(status)
    text = " ".join(
        part
        for part in (
            error_type,
            error,
            response_error,
            upstream_close_error,
            str(status_value or ""),
        )
        if part
    ).lower()
    if status_value == 429 or any(pattern in text for pattern in _RATE_LIMIT_PATTERNS):
        return UpstreamProxyFailure("llm_rate_limited", True)
    if status_value is not None and 500 <= status_value <= 599:
        return UpstreamProxyFailure("llm_upstream_unavailable", True)
    if status_value is not None and status_value >= 400:
        return UpstreamProxyFailure("llm_http_error", False)
    if response_error:
        return UpstreamProxyFailure("llm_stream_failed", True)
    if kind == "upstream_error":
        if error_type in _RETRYABLE_UPSTREAM_ERROR_TYPES or any(pattern in text for pattern in _UNAVAILABLE_PATTERNS):
            return UpstreamProxyFailure("llm_upstream_unavailable", True)
        return UpstreamProxyFailure("llm_upstream_failed", True)
    if upstream_close_error:
        return UpstreamProxyFailure("upstream_response_close_failed", False)
    return None


def gateway_failure_payload(
    failure: GatewayFailure,
    *,
    message: str,
    body: str = "",
    **details: object,
) -> dict[str, object]:
    """Build the JSON error shape returned by local gateway components."""
    payload: dict[str, object] = {
        "error": message,
        **failure.to_dict(),
        **{key: value for key, value in details.items() if value not in ("", None)},
    }
    if body:
        payload["body"] = body
    return payload


def _coerce_status(status: object) -> int | None:
    if status in (None, ""):
        return None
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def gateway_failure_affects_task(
    *,
    task_id: str,
    phase: str = "",
    failure_context: Mapping[str, object] | None = None,
) -> bool:
    """Return whether a gateway failure should be attributed to a task attempt."""
    if not task_id:
        return False
    failure_context = failure_context or {}
    reason = str(failure_context.get("reason") or "")
    if reason in TASK_AFFECTING_GATEWAY_REASONS:
        return True
    return phase in TASK_AFFECTING_GATEWAY_PHASES
