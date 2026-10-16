"""Execution context objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import getpid
from typing import Any


@dataclass(frozen=True)
class RunContext:
    run_id: str
    labels: dict[str, str] | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "run_id")

    def to_fields(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "labels": self.labels or {},
        }


@dataclass(frozen=True)
class TaskContext:
    run_id: str
    task_id: str
    version: int
    attempt_id: str
    agent_name: str
    agent_spec_name: str
    gateway_session_id: str | None = None
    sandbox_id: str | None = None
    worker_pid: int | None = None
    scope: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "run_id")
        _require_non_empty(self.task_id, "task_id")
        if self.version < 0:
            raise ValueError("version must be >= 0")
        _require_non_empty(self.attempt_id, "attempt_id")
        _require_non_empty(self.agent_name, "agent_name")
        _require_non_empty(self.agent_spec_name, "agent_spec_name")
        if self.gateway_session_id is not None:
            _require_non_empty(self.gateway_session_id, "gateway_session_id")
        if self.sandbox_id is not None:
            _require_non_empty(self.sandbox_id, "sandbox_id")
        if self.worker_pid is not None and self.worker_pid < 1:
            raise ValueError("worker_pid must be >= 1")
        _validate_scope(self.scope)

    def with_gateway(self, gateway_session_id: str) -> TaskContext:
        return TaskContext(**{**self.__dict__, "gateway_session_id": gateway_session_id})

    def with_sandbox(self, sandbox_id: str) -> TaskContext:
        return TaskContext(**{**self.__dict__, "sandbox_id": sandbox_id})

    def with_worker_pid(self) -> TaskContext:
        return TaskContext(**{**self.__dict__, "worker_pid": getpid()})

    def with_scope(self, **scope: object) -> TaskContext:
        dedicated_fields = {
            "gateway_session_id",
            "sandbox_id",
            "worker_pid",
        }
        values = dict(self.__dict__)
        next_scope = dict(self.scope)
        for key, value in scope.items():
            if key in dedicated_fields:
                values[key] = value
            else:
                next_scope[key] = value
        values["scope"] = next_scope
        return TaskContext(**values)

    def to_fields(self) -> dict[str, Any]:
        fields = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "version": self.version,
            "attempt_id": self.attempt_id,
            "agent_name": self.agent_name,
            "agent_spec_name": self.agent_spec_name,
            "gateway_session_id": self.gateway_session_id,
            "sandbox_id": self.sandbox_id,
            "worker_pid": self.worker_pid,
        }
        fields.update(self.scope)
        return fields


def _require_non_empty(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} is required")


_RESERVED_SCOPE_FIELDS = frozenset(
    {
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
)


def _validate_scope(scope: dict[str, Any]) -> None:
    for key, value in scope.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("scope keys must be non-empty strings")
        if key in _RESERVED_SCOPE_FIELDS:
            raise ValueError(f"scope cannot override reserved field {key!r}")
        if value is None:
            raise ValueError(f"scope field {key!r} cannot be None")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"scope field {key!r} cannot be empty")
