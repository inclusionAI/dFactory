"""Asynchronous scheduling of canonical execution requests."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

from anexus.core.errors import as_error_envelope
from anexus.core.execution import ExecutionRequest
from anexus.events import EventEnvelope
from anexus.tasks import TaskAttemptRecord


class AttemptRunner(Protocol):
    """Execute one fully compiled attempt request."""

    async def run(self, request: ExecutionRequest) -> TaskAttemptRecord:
        """Execute one request and return its durable result.

        Args:
            request: Compiled request containing the attempt identity and plan.

        Returns:
            The final task-attempt record.

        """
        ...


class EventPublisher(Protocol):
    """Publish one runtime event."""

    def __call__(self, event: EventEnvelope) -> None:
        """Publish an event envelope.

        Args:
            event: Event to publish.

        """
        ...


@dataclass(frozen=True)
class QueuePolicy:
    """Concurrency policy applied to compiled execution requests.

    Attributes:
        n_concurrent: Maximum requests admitted globally.
        agent_concurrency_limits: Limits keyed by agent pool name.
        agent_concurrency_groups: Mapping from agent name to shared pool name.

    """

    n_concurrent: int = 10
    agent_concurrency_limits: dict[str, int] = field(default_factory=dict)
    agent_concurrency_groups: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate global and per-agent concurrency limits."""
        if self.n_concurrent < 1:
            raise ValueError("QueuePolicy.n_concurrent must be >= 1")
        for pool_name, limit in self.agent_concurrency_limits.items():
            if not pool_name:
                raise ValueError("agent concurrency pool name is required")
            if limit < 1:
                raise ValueError(f"agent concurrency limit for {pool_name!r} must be >= 1")
            if limit > self.n_concurrent:
                raise ValueError(f"agent concurrency limit for {pool_name!r} must not exceed n_concurrent")
        for agent_name, pool_name in self.agent_concurrency_groups.items():
            if not agent_name:
                raise ValueError("agent concurrency group agent name is required")
            if not pool_name:
                raise ValueError(f"agent concurrency group for {agent_name!r} is required")
            if pool_name not in self.agent_concurrency_limits:
                raise ValueError(
                    f"agent concurrency group {pool_name!r} for {agent_name!r} requires an agent concurrency limit"
                )


class AttemptRunnerError(RuntimeError):
    """Unexpected exception raised by an attempt runner.

    Args:
        request: Request whose runner failed to return a record.
        original: Original runner exception.

    """

    def __init__(self, request: ExecutionRequest, original: Exception) -> None:
        """Attach the failed request and original runner exception."""
        self.request = request
        self.original = original
        context = request.context
        super().__init__(
            f"attempt runner failed for {context.agent_name}/{context.task_id} "
            f"({context.attempt_id}): {original or original.__class__.__name__}"
        )

    def to_event_payload(self) -> dict[str, object]:
        """Return structured request and exception context.

        Returns:
            Event-safe runner failure details.

        """
        return {
            **self.request.context.to_fields(),
            "payload_sha256": self.request.task.payload_sha256,
            "exception_type": self.__class__.__name__,
            "message": str(self),
            "original_exception_type": self.original.__class__.__name__,
            "original_message": str(self.original) or self.original.__class__.__name__,
        }


class TaskRunQueue:
    """Run compiled execution requests with global and per-agent limits.

    Args:
        policy: Queue concurrency policy.
        runner: Run-scoped attempt runner.
        event_publisher: Optional event publisher used for queue observations.

    """

    def __init__(
        self,
        *,
        policy: QueuePolicy,
        runner: AttemptRunner,
        event_publisher: EventPublisher | None = None,
    ) -> None:
        """Initialize queue policy, runner, and concurrency semaphores."""
        self.policy = policy
        self.runner = runner
        self.event_publisher = event_publisher
        self._global_semaphore = asyncio.Semaphore(policy.n_concurrent)
        self._agent_semaphores = {
            pool_name: asyncio.Semaphore(limit) for pool_name, limit in policy.agent_concurrency_limits.items()
        }

    async def run(
        self,
        requests: Iterable[ExecutionRequest],
    ) -> list[TaskAttemptRecord]:
        """Execute requests and preserve their input order in the result.

        Args:
            requests: Fully compiled requests. The queue performs no task or
                attempt expansion.

        Returns:
            Final attempt records in the same order as the requests.

        Raises:
            AttemptRunnerError: If a runner violates its contract by raising
                instead of returning a final record.
            asyncio.CancelledError: If the run is cancelled.

        """
        requests = list(requests)
        for queue_index, request in enumerate(requests):
            self.log("attempt.queued", request, queue_index)
        async with asyncio.TaskGroup() as task_group:
            futures = [
                task_group.create_task(self._run_one(request, queue_index))
                for queue_index, request in enumerate(requests)
            ]
        return [future.result() for future in futures]

    async def _run_one(
        self,
        request: ExecutionRequest,
        queue_index: int,
    ) -> TaskAttemptRecord:
        """Admit and execute one queued request.

        Args:
            request: Compiled attempt request.
            queue_index: Stable zero-based queue position.

        Returns:
            Final attempt record returned by the runner.

        Raises:
            AttemptRunnerError: If the runner raises an unexpected exception.

        """
        started = perf_counter()
        try:
            global_wait_started = perf_counter()
            async with self._global_semaphore:
                global_wait_seconds = perf_counter() - global_wait_started
                pool_name = self._agent_pool_name(request)
                semaphore = self._agent_semaphores.get(pool_name)
                agent_wait_seconds = 0.0
                if semaphore is not None:
                    agent_wait_started = perf_counter()
                    await semaphore.acquire()
                    agent_wait_seconds = perf_counter() - agent_wait_started
                try:
                    self.log(
                        "attempt.admitted",
                        request,
                        queue_index,
                        queue_wait_seconds=perf_counter() - started,
                        global_wait_seconds=global_wait_seconds,
                        agent_wait_seconds=agent_wait_seconds,
                    )
                    return await self.runner.run(request)
                finally:
                    if semaphore is not None:
                        semaphore.release()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = as_error_envelope(exc, component="runtime.queue")
            self.log(
                "attempt.runner_failed",
                request,
                queue_index,
                {**error.to_dict(), "source": "runtime.queue"},
            )
            raise AttemptRunnerError(request, exc) from exc

    def _agent_pool_name(self, request: ExecutionRequest) -> str:
        """Return the shared concurrency pool for a request.

        Args:
            request: Request containing compiled pool metadata.

        Returns:
            Agent pool name used to select a semaphore.

        """
        agent_name = request.context.agent_name
        return self.policy.agent_concurrency_groups.get(agent_name, agent_name)

    def log(
        self,
        name: str,
        request: ExecutionRequest,
        queue_index: int,
        payload: dict[str, object] | None = None,
        **fields: object,
    ) -> None:
        """Record one queue event without obscuring scheduling flow.

        Args:
            name: Stable event name.
            request: Attempt identity source.
            queue_index: Stable zero-based queue position.
            payload: Optional prebuilt event fields.
            **fields: Inline event fields for simple call sites.

        """
        if self.event_publisher is None:
            return
        queue_payload = {
            "queue_index": queue_index,
            "n_concurrent": self.policy.n_concurrent,
            "agent_pool_name": self._agent_pool_name(request),
            "agent_concurrency_limit": self.policy.agent_concurrency_limits.get(
                self._agent_pool_name(request),
                0,
            ),
        }
        self.event_publisher(
            EventEnvelope(
                name=name,
                context={
                    **request.context.to_fields(),
                    "payload_sha256": request.task.payload_sha256,
                },
                payload={**queue_payload, **(payload or {}), **fields},
            )
        )
