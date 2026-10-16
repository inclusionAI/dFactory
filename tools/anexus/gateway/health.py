"""Gateway session provider health checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from anexus.core.context import TaskContext
from anexus.core.errors import ErrorEnvelope, as_error_envelope
from anexus.core.execution import GatewayPlan
from anexus.gateway.session import GatewaySession, GatewaySessionProvider


@dataclass(frozen=True)
class GatewayHealthReport:
    """Report one gateway provider checkout and cleanup probe."""

    session_id: str
    status: Literal["healthy", "failed", "cleanup_pending"]
    latency_seconds: float
    env_keys: list[str] = field(default_factory=list)
    trace_ref_count: int = 0
    errors: list[ErrorEnvelope] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable health report."""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "latency_seconds": self.latency_seconds,
            "env_keys": list(self.env_keys),
            "trace_ref_count": self.trace_ref_count,
            "errors": [error.to_dict() for error in self.errors],
        }


async def check_gateway_session(
    gateway: GatewaySessionProvider,
    context: TaskContext,
    *,
    spec: GatewayPlan | None = None,
) -> GatewayHealthReport:
    """Checkout and release one gateway session through the runtime protocol.

    Args:
        gateway: Run-scoped gateway provider under test.
        context: Diagnostic attempt context.
        spec: Optional checkout plan. The no-gateway plan is used by default.

    Returns:
        A structured provider health report.

    """
    started = perf_counter()
    session_id = ""
    env_keys: list[str] = []
    trace_ref_count = 0
    started_provider = False
    session: GatewaySession | None = None
    try:
        await gateway.start(context)
        started_provider = True
        session = await gateway.checkout(context, spec or GatewayPlan())
        session_id = session.session_id
        env_keys = sorted(session.agent_env())
        trace_ref_count = len(session.trace_refs())
    except Exception as exc:
        errors = [as_error_envelope(exc, component="gateway.health")]
        if started_provider:
            try:
                await gateway.stop()
            except Exception as cleanup_exc:
                errors.append(as_error_envelope(cleanup_exc, component="gateway.health.cleanup"))
        return GatewayHealthReport(
            session_id=session_id,
            status="failed",
            latency_seconds=perf_counter() - started,
            env_keys=env_keys,
            trace_ref_count=trace_ref_count,
            errors=errors,
        )

    cleanup_errors: list[ErrorEnvelope] = []
    try:
        if session is not None:
            await session.release()
    except Exception as exc:
        cleanup_errors.append(as_error_envelope(exc, component="gateway.health.cleanup"))
    try:
        await gateway.stop()
    except Exception as exc:
        cleanup_errors.append(as_error_envelope(exc, component="gateway.health.cleanup"))

    if cleanup_errors:
        return GatewayHealthReport(
            session_id=session_id,
            status="cleanup_pending",
            latency_seconds=perf_counter() - started,
            env_keys=env_keys,
            trace_ref_count=trace_ref_count,
            errors=cleanup_errors,
        )

    return GatewayHealthReport(
        session_id=session_id,
        status="healthy",
        latency_seconds=perf_counter() - started,
        env_keys=env_keys,
        trace_ref_count=trace_ref_count,
    )
