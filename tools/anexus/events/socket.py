"""Unix socket transport for event-bus streams."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from anexus.events.envelope import EventEnvelope

DEFAULT_EVENT_SOCKET_PATH = "/tmp/anexus-events.sock"

EventPublisher = Callable[[EventEnvelope], None]
SocketErrorCallback = Callable[[BaseException, bytes], None]


@dataclass(frozen=True)
class SocketEventError:
    """Recoverable event socket parse or transport diagnostic."""

    exception_type: str
    message: str
    raw_line: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "exception_type": self.exception_type,
            "message": self.message,
            "raw_line": self.raw_line,
        }


class UnixEventServer:
    """Async Unix socket server that publishes newline-delimited event envelopes."""

    def __init__(
        self,
        socket_path: str | Path = DEFAULT_EVENT_SOCKET_PATH,
        *,
        event_publisher: EventPublisher | None = None,
        on_error: SocketErrorCallback | None = None,
    ):
        self._socket_path = Path(socket_path)
        self._event_publisher = event_publisher
        self._on_error = on_error
        self._server: asyncio.AbstractServer | None = None
        self._clients: set[asyncio.Task[None]] = set()
        self._errors: list[SocketEventError] = []
        self._lock = Lock()

    async def start(self) -> None:
        """Start the socket server, replacing a stale local socket path."""

        if self._server is not None:
            return
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            self._socket_path.unlink()
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._socket_path),
        )
        with contextlib.suppress(OSError):
            os.chmod(self._socket_path, 0o600)

    async def stop(self) -> None:
        """Stop the server, cancel active clients, and remove the socket file."""

        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        for task in list(self._clients):
            task.cancel()
        if self._clients:
            await asyncio.gather(*self._clients, return_exceptions=True)
        self._clients.clear()
        with contextlib.suppress(FileNotFoundError, OSError):
            self._socket_path.unlink()

    def close(self) -> None:
        """Best-effort synchronous close for shutdown paths that cannot await."""

        server = self._server
        self._server = None
        if server is not None:
            server.close()
        for task in list(self._clients):
            task.cancel()
        self._clients.clear()
        with contextlib.suppress(FileNotFoundError, OSError):
            self._socket_path.unlink()

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    def errors(self) -> list[SocketEventError]:
        with self._lock:
            return list(self._errors)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._clients.add(task)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    event = event_from_json_line(line)
                    if self._event_publisher is not None:
                        self._event_publisher(event)
                except Exception as exc:
                    self._record_error(exc, line)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            if task is not None:
                self._clients.discard(task)

    def _record_error(self, exc: BaseException, line: bytes) -> None:
        raw_line = line.decode("utf-8", errors="replace").rstrip("\n")
        error = SocketEventError(
            exception_type=exc.__class__.__name__,
            message=str(exc) or exc.__class__.__name__,
            raw_line=raw_line,
        )
        with self._lock:
            self._errors.append(error)
        if self._on_error is not None:
            with contextlib.suppress(Exception):
                self._on_error(exc, line)


class UnixEventSink:
    """Synchronous event sink that sends one event envelope to a Unix socket."""

    def __init__(
        self,
        socket_path: str | Path = DEFAULT_EVENT_SOCKET_PATH,
        *,
        timeout_seconds: float = 1.0,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        self.socket_path = Path(socket_path)
        self.timeout_seconds = timeout_seconds

    def emit(self, event: EventEnvelope) -> None:
        payload = event_to_json_line(event)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout_seconds)
            client.connect(str(self.socket_path))
            client.sendall(payload)


def event_to_json_line(event: EventEnvelope) -> bytes:
    """Serialize one event envelope as compact UTF-8 JSONL bytes."""

    return (json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def event_from_json_line(line: bytes | str) -> EventEnvelope:
    """Parse one newline-delimited serialized event envelope."""

    raw = line.decode("utf-8") if isinstance(line, bytes) else line
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("event line must decode to an object")
    return event_from_dict(data)


def event_from_dict(data: dict[str, Any]) -> EventEnvelope:
    """Build an :class:`EventEnvelope` from a JSON object."""

    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("event name is required")
    context = data.get("context")
    payload = data.get("payload", {})
    if not isinstance(context, dict):
        raise ValueError("event context must be an object")
    if not isinstance(payload, dict):
        raise ValueError("event payload must be an object")
    event_id = str(data.get("event_id", "")).strip()
    timestamp = str(data.get("timestamp", "")).strip()
    kwargs: dict[str, Any] = {
        "name": name,
        "context": dict(context),
        "payload": dict(payload),
    }
    if event_id:
        kwargs["event_id"] = event_id
    if timestamp:
        kwargs["timestamp"] = timestamp
    return EventEnvelope(**kwargs)
