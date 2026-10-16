"""In-process event bus."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol

from anexus.events.envelope import EventEnvelope


class EventSink(Protocol):
    def emit(self, event: EventEnvelope) -> None: ...


class EventBus(Protocol):
    def publish(self, event: EventEnvelope) -> None: ...

    def replay(self) -> list[EventEnvelope]: ...


@dataclass
class InMemoryEventBus:
    """A deterministic local event bus used by tests and early adapters."""

    sinks: list[EventSink] = field(default_factory=list)
    strict_sinks: bool = True

    def __post_init__(self) -> None:
        self._events: list[EventEnvelope] = []
        self._sink_errors: list[dict[str, str]] = []
        self._lock = Lock()

    def publish(self, event: EventEnvelope) -> None:
        with self._lock:
            self._events.append(event)
            sinks = list(self.sinks)
        for sink in sinks:
            try:
                sink.emit(event)
            except Exception as exc:
                if self.strict_sinks:
                    raise
                with self._lock:
                    self._sink_errors.append(
                        {
                            "event_id": event.event_id,
                            "sink": sink.__class__.__name__,
                            "exception_type": exc.__class__.__name__,
                            "message": str(exc) or exc.__class__.__name__,
                        }
                    )

    def replay(self) -> list[EventEnvelope]:
        with self._lock:
            return list(self._events)

    def sink_errors(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._sink_errors)
