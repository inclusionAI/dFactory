"""Run-scoped gateway activity polling."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from time import monotonic

from anexus.events import EventEnvelope
from anexus.gateway.session import GatewayActivitySnapshot, GatewaySessionProvider

logger = logging.getLogger(__name__)

ActivityPublisher = Callable[[EventEnvelope], None]


@dataclass
class _PublishedState:
    request_count: int = 0
    available: bool | None = None
    first_request_published: bool = False
    last_summary_at: float = 0.0


class GatewayActivityMonitor:
    """Poll gateway activity without making terminal rendering own network I/O."""

    def __init__(
        self,
        provider: GatewaySessionProvider,
        publish: ActivityPublisher,
        *,
        poll_interval: float = 2.0,
        summary_interval: float = 30.0,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        if summary_interval <= 0:
            raise ValueError("summary_interval must be > 0")
        self.provider = provider
        self.publish = publish
        self.poll_interval = poll_interval
        self.summary_interval = summary_interval
        self._latest: dict[str, GatewayActivitySnapshot] = {}
        self._published: dict[str, _PublishedState] = {}
        self._lock = Lock()
        self._stop_event: asyncio.Event | None = None
        self._stop_requested = False

    def snapshots(self) -> list[GatewayActivitySnapshot]:
        """Return the latest in-memory activity snapshots."""
        with self._lock:
            return list(self._latest.values())

    def stop(self) -> None:
        """Request that the active monitor loop stop."""
        self._stop_requested = True
        if self._stop_event is not None:
            self._stop_event.set()

    async def run(self) -> None:
        """Poll until stopped; provider failures remain diagnostic-only."""
        self._stop_event = asyncio.Event()
        if self._stop_requested:
            self._stop_event.set()
        while not self._stop_event.is_set():
            await self._poll_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval)
            except TimeoutError:
                continue

    async def _poll_once(self) -> None:
        getter = getattr(self.provider, "activity_snapshots", None)
        if not callable(getter):
            return
        try:
            snapshots = await getter()
        except Exception:
            logger.warning("gateway activity polling failed", exc_info=True)
            return

        active_ids = {snapshot.gateway_session_id for snapshot in snapshots}
        with self._lock:
            self._latest = {
                session_id: snapshot for session_id, snapshot in self._latest.items() if session_id in active_ids
            }
            self._latest.update({snapshot.gateway_session_id: snapshot for snapshot in snapshots})

        now = monotonic()
        for snapshot in snapshots:
            self._publish_if_needed(snapshot, now=now)
        for session_id in list(self._published):
            if session_id not in active_ids:
                del self._published[session_id]

    def _publish_if_needed(self, snapshot: GatewayActivitySnapshot, *, now: float) -> None:
        session_id = snapshot.gateway_session_id
        state = self._published.get(session_id)
        if state is None:
            state = _PublishedState(
                last_summary_at=now,
            )
            self._published[session_id] = state

        sample = ""
        availability_changed = state.available is not None and snapshot.available != state.available
        initial_unavailable = state.available is None and not snapshot.available
        first_request = snapshot.request_count > 0 and not state.first_request_published
        periodic = now - state.last_summary_at >= self.summary_interval
        if first_request:
            sample = "first"
            state.first_request_published = True
        elif initial_unavailable or availability_changed or periodic:
            sample = "periodic"
        if not sample:
            return

        delta = max(snapshot.request_count - state.request_count, 0)
        event = EventEnvelope(
            name="gateway.activity",
            context=snapshot.event_context(),
            payload=snapshot.event_payload(sample=sample, request_delta=delta),
        )
        try:
            self.publish(event)
        except Exception:
            logger.warning("gateway activity event publishing failed", exc_info=True)
            return
        if availability_changed and not snapshot.available:
            logger.warning(
                "gateway activity unavailable for session %s: %s",
                session_id,
                snapshot.error or snapshot.error_type or "unknown error",
            )
        state.request_count = snapshot.request_count
        state.available = snapshot.available
        state.last_summary_at = now
