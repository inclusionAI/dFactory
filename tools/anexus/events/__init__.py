"""Typed event envelopes and bus implementations."""

from anexus.events.bus import EventBus, InMemoryEventBus
from anexus.events.envelope import EventEnvelope
from anexus.events.reducers import AttemptSnapshot, SidecarSnapshot, reduce_attempts
from anexus.events.sinks import JsonlEventSink
from anexus.events.socket import (
    DEFAULT_EVENT_SOCKET_PATH,
    SocketEventError,
    UnixEventServer,
    UnixEventSink,
    event_from_dict,
    event_from_json_line,
    event_to_json_line,
)

__all__ = [
    "AttemptSnapshot",
    "EventBus",
    "EventEnvelope",
    "InMemoryEventBus",
    "JsonlEventSink",
    "SidecarSnapshot",
    "DEFAULT_EVENT_SOCKET_PATH",
    "SocketEventError",
    "UnixEventServer",
    "UnixEventSink",
    "event_from_dict",
    "event_from_json_line",
    "event_to_json_line",
    "reduce_attempts",
]
