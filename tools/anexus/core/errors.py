"""Error contracts used across process and provider boundaries."""

from __future__ import annotations

from asyncio import CancelledError
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self


class ErrorKind(StrEnum):
    """Stable error classes for scheduler decisions."""

    AGENT_ERROR = "agent_error"
    ARTIFACT_ERROR = "artifact_error"
    CANCELLED = "cancelled"
    GATEWAY_ERROR = "gateway_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    SANDBOX_ERROR = "sandbox_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ErrorEnvelope:
    """Serializable error detail for logs, events, and attempt records."""

    kind: ErrorKind
    message: str
    component: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        default_component: str = "",
    ) -> Self:
        """Restore a serialized envelope, tolerating unknown error kinds."""

        try:
            kind = ErrorKind(str(data.get("kind", ErrorKind.UNKNOWN.value)))
        except ValueError:
            kind = ErrorKind.UNKNOWN
        details = data.get("details")
        return cls(
            kind=kind,
            message=str(data.get("message", "")),
            component=str(data.get("component", default_component)),
            retryable=bool(data.get("retryable", False)),
            details=dict(details) if isinstance(details, Mapping) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "message": self.message,
            "component": self.component,
            "retryable": self.retryable,
            "details": self.details,
        }


class AnexusError(RuntimeError):
    """Operational failure carrying scheduler-relevant metadata."""

    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind,
        component: str,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.component = component
        self.retryable = retryable
        self.details = dict(details or {})


def as_error_envelope(exc: BaseException, *, component: str) -> ErrorEnvelope:
    if isinstance(exc, AnexusError):
        return ErrorEnvelope(
            kind=exc.kind,
            message=str(exc),
            component=exc.component,
            retryable=exc.retryable,
            details=dict(exc.details),
        )
    if isinstance(exc, (KeyboardInterrupt, CancelledError)):
        return ErrorEnvelope(
            kind=ErrorKind.CANCELLED,
            message=str(exc) or exc.__class__.__name__,
            component=component,
            details={"exception_type": exc.__class__.__name__},
        )
    if isinstance(exc, TimeoutError):
        return ErrorEnvelope(
            kind=ErrorKind.TIMEOUT,
            message=str(exc) or exc.__class__.__name__,
            component=component,
            retryable=True,
            details={"exception_type": exc.__class__.__name__},
        )
    return ErrorEnvelope(
        kind=ErrorKind.UNKNOWN,
        message=str(exc) or exc.__class__.__name__,
        component=component,
        details={"exception_type": exc.__class__.__name__},
    )
