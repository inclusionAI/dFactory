"""Typed event envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from anexus.core.context import TaskContext
from anexus.core.ids import new_id


@dataclass(frozen=True)
class EventEnvelope:
    name: str
    context: dict[str, Any]
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: new_id("evt"))
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def for_task(
        cls,
        name: str,
        context: TaskContext,
        payload: dict[str, Any] | None = None,
    ) -> EventEnvelope:
        return cls(name=name, context=context.to_fields(), payload=payload or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "name": self.name,
            "context": self.context,
            "payload": self.payload,
        }
