"""Sandbox provider health checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from anexus.core.context import TaskContext
from anexus.core.errors import ErrorEnvelope, as_error_envelope
from anexus.sandbox.provider import (
    SandboxLaunchSpec,
    SandboxLease,
    SandboxProvider,
)


@dataclass(frozen=True)
class SandboxHealthReport:
    provider: str
    status: Literal["healthy", "failed", "cleanup_pending"]
    latency_seconds: float
    sandbox_id: str | None = None
    errors: list[ErrorEnvelope] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "status": self.status,
            "latency_seconds": self.latency_seconds,
            "sandbox_id": self.sandbox_id,
            "errors": [error.to_dict() for error in self.errors],
        }


async def check_sandbox_provider(
    provider: SandboxProvider,
    context: TaskContext,
    spec: SandboxLaunchSpec | None = None,
) -> SandboxHealthReport:
    """Create and delete a sandbox to validate provider availability."""

    started = perf_counter()
    lease: SandboxLease | None = None
    errors: list[ErrorEnvelope] = []
    try:
        lease = await provider.create(context, spec)
    except Exception as exc:
        return SandboxHealthReport(
            provider=provider.provider,
            status="failed",
            latency_seconds=perf_counter() - started,
            errors=[as_error_envelope(exc, component="sandbox.health")],
        )

    try:
        await provider.delete(lease)
    except Exception as exc:
        errors.append(as_error_envelope(exc, component="sandbox.health"))
        return SandboxHealthReport(
            provider=lease.provider,
            status="cleanup_pending",
            latency_seconds=perf_counter() - started,
            sandbox_id=lease.sandbox_id,
            errors=errors,
        )

    return SandboxHealthReport(
        provider=lease.provider,
        status="healthy",
        latency_seconds=perf_counter() - started,
        sandbox_id=lease.sandbox_id,
    )
