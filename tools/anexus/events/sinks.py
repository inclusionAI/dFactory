"""Event sinks used by journals and local tooling."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from anexus.core.files import append_jsonl, read_jsonl_objects_recoverable
from anexus.events.envelope import EventEnvelope


class JsonlEventSink:
    """Append events to a local JSONL file with process and file locking."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()

    def emit(self, event: EventEnvelope) -> None:
        with self._lock:
            append_jsonl(self.path, event.to_dict())

    def replay(self) -> list[dict[str, Any]]:
        rows, _errors = read_jsonl_objects_recoverable(self.path)
        return rows
