"""Run-level controller."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter

from anexus.config import RunPlan
from anexus.core.errors import as_error_envelope
from anexus.events import EventBus, EventEnvelope
from anexus.runtime.planning import (
    ExecutionRequestCompiler,
    ExecutionSchedule,
    select_logical_records,
)
from anexus.runtime.queue import AttemptRunner, AttemptRunnerError, QueuePolicy, TaskRunQueue
from anexus.runtime.run_lock import RunLock, build_run_lock, require_matching_run_lock
from anexus.runtime.store import LocalRunStore, record_event_safely
from anexus.sandbox.resources import read_sandbox_provider_resource_snapshots
from anexus.tasks import TaskAttemptRecord


@dataclass(frozen=True)
class RunResult:
    """Aggregate operational and evaluation outcomes for one run."""

    run_id: str
    agent_names: tuple[str, ...]
    status: str
    records: list[TaskAttemptRecord]
    timings: dict[str, float]
    status_counts: dict[str, int] = field(default_factory=dict)
    evaluation_counts: dict[str, int] = field(default_factory=dict)
    total_slot_count: int = 0
    resumed_slot_count: int = 0
    scheduled_slot_count: int = 0
    executed_attempt_count: int = 0
    physical_attempt_count: int = 0


@dataclass(frozen=True)
class PreparedRun:
    """Validated RunLock and resume-aware schedule before provider creation."""

    run_lock: RunLock
    schedule: ExecutionSchedule


class RunController:
    """Execute a resolved run plan through the runtime queue."""

    def __init__(
        self,
        *,
        bus: EventBus,
        run_store: LocalRunStore | None = None,
        resume_completed: bool = False,
        trace_storage: Mapping[str, object] | None = None,
    ):
        """Initialize a run controller.

        Args:
            bus: Event bus used for run and attempt observations.
            run_store: Optional durable run store.
            resume_completed: Whether completed attempts should be omitted when
                compiling execution requests.

        """
        self.bus = bus
        self.run_store = run_store
        self.resume_completed = resume_completed
        self.trace_storage = dict(trace_storage or {})
        self.request_compiler = ExecutionRequestCompiler(
            run_store=run_store,
            resume_completed=resume_completed,
        )

    def run(
        self,
        plan: RunPlan,
        *,
        runner: AttemptRunner,
        sandbox_provider: object | None = None,
    ) -> RunResult:
        """Execute a run plan in a new event loop.

        Args:
            plan: Fully resolved run plan.
            runner: Run-scoped executor for canonical execution requests.
            sandbox_provider: Optional provider used for resource snapshots.

        Returns:
            Aggregated run result.

        """
        return asyncio.run(
            self.run_async(
                plan,
                runner=runner,
                sandbox_provider=sandbox_provider,
            )
        )

    async def run_async(
        self,
        plan: RunPlan,
        *,
        runner: AttemptRunner | None,
        sandbox_provider: object | None = None,
        prepared: PreparedRun | None = None,
    ) -> RunResult:
        """Compile and execute all requests for a run plan.

        Args:
            plan: Fully resolved run plan.
            runner: Run-scoped executor for canonical execution requests.
            sandbox_provider: Optional provider used for resource snapshots.

        Returns:
            Aggregated run result.

        """
        prepared = prepared or self.prepare(plan)
        run_lock = prepared.run_lock
        schedule = prepared.schedule
        requests = schedule.requests
        started = perf_counter()
        self.log(
            "run.started",
            plan,
            task_count=len(plan.tasks),
            agent_names=list(plan.agent_names),
            request_count=len(requests),
            n_concurrent=plan.n_concurrent,
            n_attempts=plan.n_attempts,
            agent_concurrency_limits=dict(plan.agent_concurrency_limits),
            agent_concurrency_groups=dict(plan.agent_concurrency_groups),
            resume_completed=self.resume_completed,
            fingerprint=run_lock.fingerprint,
            total_slot_count=schedule.total_slot_count,
            resumed_slot_count=schedule.resumed_slot_count,
            scheduled_slot_count=schedule.scheduled_slot_count,
            physical_attempt_count=schedule.physical_attempt_count,
        )
        if requests and runner is None:
            raise ValueError("attempt runner is required when the run has pending slots")
        try:
            if requests:
                await self._publish_sandbox_resource_snapshot(plan, sandbox_provider, phase="run.start")
                queue = TaskRunQueue(
                    policy=QueuePolicy(
                        n_concurrent=plan.n_concurrent,
                        agent_concurrency_limits=dict(plan.agent_concurrency_limits),
                        agent_concurrency_groups=dict(plan.agent_concurrency_groups),
                    ),
                    runner=runner,
                    event_publisher=self._publish,
                )
                executed_records = await queue.run(requests)
            else:
                executed_records = []
        except BaseException as exc:
            await self._publish_sandbox_resource_snapshot(plan, sandbox_provider, phase="run.end")
            self.log(
                "run.cancelled" if _is_cancelled_exception(exc) else "run.failed",
                plan,
                _exception_payload(exc),
            )
            raise

        stored_attempts = self.run_store.list_attempts(plan.run_id) if self.run_store else []
        records, physical_attempt_count = select_logical_records(
            plan,
            stored_attempts,
            executed=executed_records,
        )
        status_counts = _status_counts(records)
        status = _run_status(records)
        result = RunResult(
            run_id=plan.run_id,
            agent_names=plan.agent_names,
            status=status,
            records=records,
            timings={"run_seconds": perf_counter() - started},
            status_counts=status_counts,
            evaluation_counts=_evaluation_counts(records),
            total_slot_count=schedule.total_slot_count,
            resumed_slot_count=schedule.resumed_slot_count,
            scheduled_slot_count=schedule.scheduled_slot_count,
            executed_attempt_count=len(executed_records),
            physical_attempt_count=physical_attempt_count,
        )
        if requests:
            await self._publish_sandbox_resource_snapshot(plan, sandbox_provider, phase="run.end")
        self.log(_run_terminal_event_name(result.status), plan, _run_result_payload(result))
        return result

    def prepare(self, plan: RunPlan) -> PreparedRun:
        """Validate resume state and persist intents before provider allocation."""
        run_lock = self._claim_run_lock(plan)
        schedule = self.request_compiler.compile_schedule(plan)
        if self.run_store is not None:
            for request in schedule.requests:
                context = request.context
                intent = {
                    "schema_version": 1,
                    **context.to_fields(),
                    "payload_sha256": request.task.payload_sha256,
                    "created_at": datetime.now(UTC).isoformat(),
                }
                path = self.run_store.write_attempt_intent(intent)
                self._publish(
                    EventEnvelope.for_task(
                        "attempt.intent_created",
                        context,
                        {
                            "path": str(path),
                            "payload_sha256": request.task.payload_sha256,
                        },
                    )
                )
        return PreparedRun(run_lock=run_lock, schedule=schedule)

    def log(
        self,
        name: str,
        plan: RunPlan,
        payload: dict[str, object] | None = None,
        **fields: object,
    ) -> None:
        """Record one run event without obscuring controller flow.

        Args:
            name: Stable event name.
            plan: Run identity source.
            payload: Optional prebuilt event fields.
            **fields: Inline event fields for simple call sites.

        """
        self._publish(
            EventEnvelope(
                name=name,
                context=_run_context(plan),
                payload={**(payload or {}), **fields},
            )
        )

    def _publish(self, event: EventEnvelope) -> None:
        self.bus.publish(event)
        if self.run_store is not None:
            record_event_safely(self.run_store, event, self.bus.publish)

    async def _publish_sandbox_resource_snapshot(
        self,
        plan: RunPlan,
        sandbox_provider: object | None,
        *,
        phase: str,
    ) -> None:
        if sandbox_provider is None:
            return

        snapshots_fn = getattr(sandbox_provider, "resource_snapshots", None)
        snapshot_fn = getattr(sandbox_provider, "resource_snapshot", None)
        if not callable(snapshots_fn) and not callable(snapshot_fn):
            return

        try:
            snapshots = await read_sandbox_provider_resource_snapshots(sandbox_provider)
        except Exception as exc:
            error = as_error_envelope(exc, component="runtime.resource_snapshot")
            self.log(
                "sandbox.resources.snapshot_failed",
                plan,
                {**error.to_dict(), "phase": phase},
            )
            return
        for snapshot in snapshots:
            self.log(
                "sandbox.resources.snapshot",
                plan,
                {**snapshot.to_summary(), "phase": phase},
            )

    def _claim_run_lock(self, plan: RunPlan) -> RunLock:
        """Create a new lock or validate the existing v4 lock for resume."""
        current = build_run_lock(plan, trace_storage=self.trace_storage)
        if self.run_store is None:
            if self.resume_completed:
                raise ValueError("resume requires a durable run store")
            return current
        stored = self.run_store.claim_run_lock(
            current.to_dict(),
            resume=self.resume_completed,
        )
        existing = RunLock.from_dict(stored)
        require_matching_run_lock(existing, current)
        return current


def _run_context(plan: RunPlan) -> dict[str, object]:
    return {"run_id": plan.run_id}


def _status_counts(records: list[TaskAttemptRecord]) -> dict[str, int]:
    """Count independent operational statuses."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    return counts


def _evaluation_counts(records: list[TaskAttemptRecord]) -> dict[str, int]:
    """Count evaluation results independently from operational status."""
    counts = {"passed": 0, "failed": 0, "not_run": 0}
    for record in records:
        if record.result is None or record.status in {"failed", "timeout", "cancelled"}:
            counts["not_run"] += 1
        elif record.result.passed:
            counts["passed"] += 1
        else:
            counts["failed"] += 1
    return counts


def _run_status(records: list[TaskAttemptRecord]) -> str:
    """Reduce attempt statuses into one run-level terminal status."""
    if not records:
        return "completed"

    statuses = {record.status for record in records}
    if statuses == {"completed"}:
        return "completed"
    if "cancelled" in statuses:
        return "cancelled"
    if statuses <= {"completed", "artifact_pending"}:
        return "artifact_pending"
    if statuses <= {"completed", "cleanup_pending"}:
        return "cleanup_pending"
    if statuses == {"timeout"}:
        return "timeout"
    if statuses <= {"failed", "timeout"}:
        return "failed"
    return "partial"


def _run_terminal_event_name(status: str) -> str:
    if status == "cancelled":
        return "run.cancelled"
    if status in {"failed", "timeout"}:
        return "run.failed"
    return "run.completed"


def _run_result_payload(result: RunResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": result.status,
        "agent_names": list(result.agent_names),
        "record_count": len(result.records),
        "status_counts": result.status_counts,
        "evaluation_counts": result.evaluation_counts,
        "total_slot_count": result.total_slot_count,
        "resumed_slot_count": result.resumed_slot_count,
        "scheduled_slot_count": result.scheduled_slot_count,
        "executed_attempt_count": result.executed_attempt_count,
        "physical_attempt_count": result.physical_attempt_count,
    }
    if result.status in {"cancelled", "failed", "timeout"}:
        payload["kind"] = result.status
        payload["errors"] = _run_error_items(result.records)
    return payload


def _run_error_items(records: list[TaskAttemptRecord]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for record in records:
        if not record.errors:
            continue
        error = record.errors[0]
        items.append(
            {
                "run_id": record.run_id,
                "task_id": record.task_id,
                "attempt_id": record.attempt_id,
                "agent_name": record.agent_name,
                "payload_sha256": record.payload_sha256,
                "status": record.status,
                "sample_index": record.sample_index,
                "execution_index": record.execution_index,
                "quality_tags": list(record.quality_tags),
                **error.to_dict(),
            }
        )
    return items


def _exception_payload(exc: BaseException) -> dict[str, object]:
    payload: dict[str, object] = {
        "exception_type": exc.__class__.__name__,
        "message": str(exc) or exc.__class__.__name__,
    }
    if _is_cancelled_exception(exc):
        payload["kind"] = "cancelled"
    if isinstance(exc, BaseExceptionGroup):
        payload["errors"] = [_exception_item(child) for child in exc.exceptions]
    return payload


def _exception_item(exc: BaseException) -> dict[str, object]:
    if isinstance(exc, AttemptRunnerError):
        return exc.to_event_payload()
    return {
        "exception_type": exc.__class__.__name__,
        "message": str(exc) or exc.__class__.__name__,
    }


def _is_cancelled_exception(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_cancelled_exception(child) for child in exc.exceptions)
    return False
