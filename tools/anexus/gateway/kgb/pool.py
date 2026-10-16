"""KGB gateway service pool.

The pool manages KGB service tickets, not sandbox slots. Sandbox lifecycle is
delegated to the existing `SandboxProvider` protocol through `KGBInstance`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from anexus.core.context import TaskContext
from anexus.core.ids import new_id
from anexus.gateway.kgb.instance import (
    KGBConfig,
    KGBInstance,
    _make_kgb_key_name,
)
from anexus.gateway.session import (
    GatewayActivitySnapshot,
    KGBSessionConfig,
    gateway_session_error,
)
from anexus.sandbox.provider import SandboxLease, SandboxProvider

logger = logging.getLogger(__name__)

TraceExportPolicy = Literal["none", "on_release", "on_retire"]


@dataclass
class KGBServiceRecord:
    """Attempt service record held by one KGB instance.

    Attributes:
        session_id: Attempt-scoped gateway session id.
        agent_name: Agent consuming this service ticket.
        task_id: Task consuming this service ticket.
        api_key_name: KGB API key name used for trace grouping.
        trace_file: Expected JSONL trace file name.
        context: Attempt context associated with this ticket.
        trace_exported: Whether an on-release trace export completed.
        release_complete: Whether every release side effect completed.
        terminal_release_error: Non-retryable release failure after the KGB
            instance has already been retired.

    """

    session_id: str
    agent_name: str
    task_id: str
    api_key_name: str
    trace_file: str
    context: TaskContext
    trace_exported: bool = False
    release_complete: bool = False
    terminal_release_error: BaseException | None = None


@dataclass
class KGBInstanceState:
    """Pool state for one live KGB sandbox."""

    instance: KGBInstance
    max_active_sessions: int
    max_lifetime_sessions: int = 1
    issued_count: int = 0
    active: dict[str, KGBServiceRecord] = field(default_factory=dict)
    completed: list[KGBServiceRecord] = field(default_factory=list)
    draining: bool = False
    retiring: bool = False
    retire_error: BaseException | None = None

    @property
    def lease(self) -> SandboxLease:
        """Return the sandbox lease owned by this KGB instance."""
        return self.instance.lease

    @property
    def available_slots(self) -> int:
        """Return currently borrowable concurrent session slots."""
        if self.draining or self.retiring:
            return 0
        return max(self.max_active_sessions - len(self.active), 0)

    @property
    def lifetime_exhausted(self) -> bool:
        """Return whether the configured rotation threshold was reached."""
        return self.issued_count >= self.max_lifetime_sessions


class KGBGatewaySession:
    """Attempt-scoped gateway session backed by one KGB service ticket."""

    def __init__(
        self,
        *,
        pool: KGBPool,
        state: KGBInstanceState,
        record: KGBServiceRecord,
        api_key: str,
    ) -> None:
        """Initialize a KGB gateway session.

        Args:
            pool: Owning KGB service pool.
            state: KGB instance state that issued the service ticket.
            record: Attempt service record.
            api_key: KGB API key value injected into the agent sandbox.

        """
        self._pool = pool
        self._state = state
        self._record = record
        self._api_key = api_key
        self._release_task: asyncio.Task[list[str]] | None = None
        self._export_traces_requested = False
        self._released = False
        self.session_id = record.session_id

    @property
    def record(self) -> KGBServiceRecord:
        """Return this session's service record."""
        return self._record

    def agent_env(self) -> dict[str, str]:
        """Return KGB environment variables for the agent sandbox."""
        return self._state.instance.build_agent_env(
            self._record.agent_name,
            self._record.task_id,
            api_key=self._api_key,
        )

    def trace_refs(self) -> list[dict[str, str]]:
        """Return trace references for this session's API key."""
        refs = self._state.instance.trace_refs_for_task(self._record.task_id)
        selected = [dict(ref) for ref in refs if ref.get("api_key_name") == self._record.api_key_name]
        if selected:
            return selected
        return [
            {
                "api_key_name": self._record.api_key_name,
                "trace_file": self._record.trace_file,
                "role": "worker",
            }
        ]

    async def activity_snapshot(self) -> GatewayActivitySnapshot | None:
        """Return the latest KGB request activity for this session."""
        return await self._pool.activity_snapshot(self)

    async def release(
        self,
        *,
        export_traces: bool = False,
        graceful: bool = True,
    ) -> list[str]:
        """Release this session's one KGB service ticket.

        Args:
            export_traces: Whether to force export of this task's trace during
                release.
            graceful: Whether optional trace export should run before the KGB
                instance is retired.

        Returns:
            Trace file names uploaded during this release.

        """
        if self._released:
            return []
        self._export_traces_requested = self._export_traces_requested or export_traces
        release_task = self._release_task
        if release_task is None:
            release_task = asyncio.create_task(
                self._pool.release(
                    self,
                    export_traces=export_traces,
                    graceful=graceful,
                )
            )
            self._release_task = release_task

        try:
            uploaded = await asyncio.shield(release_task)
        except asyncio.CancelledError:
            release_task.add_done_callback(self._observe_release_task)
            raise
        except BaseException:
            if self._release_task is release_task:
                self._release_task = None
            raise

        self._released = True
        if self._release_task is release_task:
            self._release_task = None
        return uploaded

    def _observe_release_task(self, task: asyncio.Task[list[str]]) -> None:
        """Record a shielded release result after caller cancellation.

        Args:
            task: Background release task protected from caller cancellation.

        """
        if task.cancelled():
            if self._release_task is task:
                self._release_task = None
            return
        error = task.exception()
        if error is None:
            self._released = True
        elif self._release_task is task:
            self._release_task = None


class KGBPool:
    """Run-scoped KGB service-ticket pool.

    Active session capacity and lifetime rotation are independent. The pool
    does not schedule attempts or reserve sandbox slots.
    """

    def __init__(
        self,
        config: KGBSessionConfig,
        *,
        sandbox_provider: SandboxProvider,
    ) -> None:
        """Initialize the pool.

        Args:
            config: Gateway/KGB pool configuration.
            sandbox_provider: Shared run-scoped sandbox provider.

        """
        _validate_config(config)
        self.config = config
        self._sandbox_provider = sandbox_provider
        self._condition = asyncio.Condition()
        self._states: list[KGBInstanceState] = []
        self._creating = 0
        self._started = False
        self._stopping = False
        self._context: TaskContext | None = None

    async def start(self, context: TaskContext) -> None:
        """Start the run-scoped pool and prefetch warm KGB instances."""
        async with self._condition:
            self._started = True
            self._stopping = False
            self._context = context
        if self.config.pool.prefetch_instances:
            await self._prefetch_best_effort(context)

    async def checkout(
        self,
        context: TaskContext,
        *,
        agent_name: str,
        task_id: str,
    ) -> KGBGatewaySession:
        """Checkout one KGB service ticket for an attempt."""
        await self._ensure_started(context)
        while True:
            async with self._condition:
                if self._stopping:
                    raise RuntimeError("KGB pool is stopping")
                state = self._ready_state_locked()
                if state is not None:
                    record = _service_record(
                        context=context,
                        agent_name=agent_name,
                        task_id=task_id,
                    )
                    state.issued_count += 1
                    if state.lifetime_exhausted:
                        state.draining = True
                    state.active[record.session_id] = record
                    break

                if self._can_create_locked():
                    self._creating += 1
                    create_context = self._context or context
                else:
                    blocked_error = self._retire_error_locked()
                    if blocked_error is not None:
                        raise gateway_session_error(
                            blocked_error,
                            phase="checkout",
                            context=context,
                        )
                    await self._condition.wait()
                    continue

            await _await_cleanup(self._create_state_into_pool(create_context))

        try:
            api_key = await asyncio.to_thread(
                state.instance.create_api_key,
                agent_name,
                task_id,
                "worker",
                identity=_key_identity(context, record.session_id),
            )
        except asyncio.CancelledError:
            await self._undo_checkout(state, record)
            raise
        except Exception as exc:
            await self._undo_checkout(state, record)
            raise gateway_session_error(exc, phase="checkout", context=context) from exc

        return KGBGatewaySession(
            pool=self,
            state=state,
            record=record,
            api_key=api_key,
        )

    async def release(
        self,
        session: KGBGatewaySession,
        *,
        export_traces: bool = False,
        graceful: bool = True,
    ) -> list[str]:
        """Release one KGB service ticket and retire drained instances."""
        state = session._state
        record = session._record
        async with self._condition:
            current = state.active.pop(record.session_id, None)
            if current is not None:
                state.completed.append(current)
                record = current
            else:
                completed = next(
                    (item for item in state.completed if item.session_id == record.session_id),
                    None,
                )
                if completed is None:
                    return []
                record = completed
            if record.release_complete:
                return []
            if record.terminal_release_error is not None:
                raise record.terminal_release_error
            retire = state.draining and not state.active and state in self._states
            if retire:
                state.retiring = True
            self._condition.notify_all()

        uploaded: list[str] = []
        release_errors: list[Exception] = []
        export_on_release = graceful and (
            export_traces or session._export_traces_requested or self._trace_export_policy == "on_release"
        )
        if export_on_release and not record.trace_exported:
            try:
                uploaded.extend(
                    await self._export_task_traces(
                        state,
                        [record.trace_file],
                        context=record.context,
                    )
                )
            except Exception as exc:
                release_errors.append(exc)
            else:
                record.trace_exported = True

        # A lifetime-exhausted instance must retire once its active set is
        # empty, even when an on-release trace export already failed. Retiring
        # is the hard capacity boundary; export errors remain observable below.
        if retire:
            try:
                uploaded.extend(
                    await _await_cleanup(
                        self._retire_state(
                            state,
                            context=record.context,
                            export_traces=graceful,
                        )
                    )
                )
            except Exception as exc:
                release_errors.append(exc)
                if state not in self._states:
                    record.terminal_release_error = exc
            finally:
                await self._prefetch_after_retire()

            # Once retirement succeeds, a prior on-release export failure can
            # no longer be retried against this instance.
            if release_errors and state not in self._states:
                record.terminal_release_error = release_errors[0]

        if release_errors:
            raise release_errors[0]
        record.release_complete = True
        return uploaded

    async def stop(self, *, graceful: bool = True) -> None:
        """Retire all live KGB instances and stop the pool."""
        async with self._condition:
            self._stopping = True
            while self._creating:
                await self._condition.wait()
            states = list(self._states)
            for state in states:
                state.draining = True
                state.retiring = True
            self._condition.notify_all()

        retire_tasks = [
            asyncio.create_task(
                self._retire_state(
                    state,
                    remove=True,
                    context=self._context,
                    export_traces=graceful,
                )
            )
            for state in states
        ]
        retire_results = await _await_cleanup(asyncio.gather(*retire_tasks, return_exceptions=True))
        retire_errors = [result for result in retire_results if isinstance(result, BaseException)]
        if retire_errors:
            raise retire_errors[0]
        async with self._condition:
            self._started = False

    def trace_viewer_url(self) -> str:
        """Return the configured trace viewer URL, if any."""
        return self.config.trace_viewer_url()

    def kgb_log_count(self) -> int:
        """Return the total KGB log count across live instances."""
        return sum(state.instance.log_count() for state in self._states)

    def kgb_log_counts_by_task(self, task_ids: Iterable[str]) -> dict[str, int]:
        """Return KGB log counts grouped by task id."""
        counts = {task_id: 0 for task_id in task_ids}
        for state in self._states:
            for task_id, count in state.instance.log_counts_by_task(counts.keys()).items():
                counts[task_id] = counts.get(task_id, 0) + count
        return {task_id: count for task_id, count in counts.items() if count}

    async def activity_snapshots(self) -> list[GatewayActivitySnapshot]:
        """Return request activity for all currently active sessions."""
        async with self._condition:
            active = [(state, record) for state in self._states for record in state.active.values()]
        if not active:
            return []
        return list(await asyncio.gather(*(self._activity_snapshot(state, record) for state, record in active)))

    async def activity_snapshot(
        self,
        session: KGBGatewaySession,
    ) -> GatewayActivitySnapshot:
        """Return request activity for one checked-out session."""
        return await self._activity_snapshot(session._state, session._record)

    async def _activity_snapshot(
        self,
        state: KGBInstanceState,
        record: KGBServiceRecord,
    ) -> GatewayActivitySnapshot:
        try:
            data = await asyncio.to_thread(
                state.instance.request_activity,
                record.api_key_name,
                timeout=2.0,
            )
        except Exception as exc:
            return GatewayActivitySnapshot(
                context=record.context,
                gateway_session_id=record.session_id,
                api_key_name=record.api_key_name,
                available=False,
                error_type=exc.__class__.__name__,
                error=str(exc) or exc.__class__.__name__,
            )
        return GatewayActivitySnapshot(
            context=record.context,
            gateway_session_id=record.session_id,
            api_key_name=str(data.get("api_key_name") or record.api_key_name),
            api_key_id=_optional_int(data.get("api_key_id")),
            request_count=_optional_int(data.get("request_count")) or 0,
            last_request_time=str(data.get("last_request_time") or ""),
            response_status=_optional_int(data.get("response_status")),
            first_byte_delay_ms=_optional_int(data.get("first_byte_delay_ms")),
            total_time_ms=_optional_int(data.get("total_time_ms")),
            retry_count=_optional_int(data.get("retry_count")) or 0,
            input_tokens=_optional_int(data.get("input_tokens")),
            output_tokens=_optional_int(data.get("output_tokens")),
        )

    @property
    def _trace_export_policy(self) -> TraceExportPolicy:
        return self.config.gateway.trace.export_policy  # type: ignore[return-value]

    async def _ensure_started(self, context: TaskContext) -> None:
        async with self._condition:
            if self._started:
                return
        await self.start(context)

    async def _prefetch_best_effort(self, context: TaskContext) -> None:
        target = min(
            self.config.pool.prefetch_instances,
            self.config.pool.max_live_instances,
        )
        while True:
            async with self._condition:
                if self._stopping:
                    return
                non_draining = sum(1 for state in self._states if not state.draining and not state.retiring)
                if non_draining + self._creating >= target or not self._can_create_locked():
                    return
                self._creating += 1
            try:
                await self._create_state_into_pool(context)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("KGB prefetch failed", exc_info=True)
                return

    async def _prefetch_after_retire(self) -> None:
        context = self._context
        if context is None or not self.config.pool.prefetch_instances:
            return
        await self._prefetch_best_effort(context)

    async def _create_state_into_pool(self, context: TaskContext) -> None:
        try:
            state = await self._create_state(context)
        except BaseException:
            async with self._condition:
                self._creating = max(0, self._creating - 1)
                self._condition.notify_all()
            raise

        async with self._condition:
            self._creating = max(0, self._creating - 1)
            self._states.append(state)
            if self._stopping:
                state.draining = True
                state.retiring = True
            self._condition.notify_all()

    async def _create_state(self, context: TaskContext) -> KGBInstanceState:
        instance = KGBInstance(
            _kgb_config(self.config),
            sandbox_provider=self._sandbox_provider,
        )
        try:
            await instance.start(context)
            await asyncio.to_thread(instance.configure_providers)
        except asyncio.CancelledError:
            await _await_cleanup(instance.stop())
            raise
        except Exception as exc:
            try:
                await _await_cleanup(instance.stop())
            except Exception as cleanup_exc:
                exc.add_note(f"KGB create cleanup also failed: {cleanup_exc}")
            raise gateway_session_error(exc, phase="create", context=context) from exc
        return KGBInstanceState(
            instance=instance,
            max_active_sessions=self.config.pool.max_active_sessions_per_instance,
            max_lifetime_sessions=self.config.pool.max_sessions_per_instance_lifetime,
        )

    async def _undo_checkout(self, state: KGBInstanceState, record: KGBServiceRecord) -> None:
        async with self._condition:
            state.active.pop(record.session_id, None)
            state.issued_count = max(0, state.issued_count - 1)
            if not state.lifetime_exhausted and not self._stopping:
                state.draining = False
            self._condition.notify_all()

    async def _export_task_traces(
        self,
        state: KGBInstanceState,
        trace_files: list[str] | None = None,
        *,
        context: TaskContext | None = None,
    ) -> list[str]:
        try:
            return await state.instance.export_task_traces(trace_files)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "KGB trace export failed: sandbox_id=%s",
                state.instance.sandbox_id,
                exc_info=True,
            )
            raise gateway_session_error(
                exc,
                phase="trace_export",
                context=context or self._context or _pool_context(),
            ) from exc

    async def _retire_state(
        self,
        state: KGBInstanceState,
        *,
        remove: bool = True,
        context: TaskContext | None = None,
        export_traces: bool = True,
    ) -> list[str]:
        uploaded: list[str] = []
        export_error: Exception | None = None
        if export_traces and self._trace_export_policy == "on_retire":
            try:
                uploaded.extend(await self._export_task_traces(state, context=context))
            except Exception as exc:
                export_error = exc
        try:
            await state.instance.stop()
        except Exception as exc:
            state.retire_error = gateway_session_error(
                exc,
                phase="retire",
                context=context or self._context or _pool_context(),
            )
            async with self._condition:
                self._condition.notify_all()
            raise state.retire_error from exc
        if remove:
            async with self._condition:
                if state in self._states:
                    self._states.remove(state)
                self._condition.notify_all()
        if export_error is not None:
            raise export_error
        return uploaded

    def _ready_state_locked(self) -> KGBInstanceState | None:
        ready = [state for state in self._states if state.available_slots > 0]
        if not ready:
            return None
        return max(ready, key=lambda state: state.available_slots)

    def _can_create_locked(self) -> bool:
        return not self._stopping and len(self._states) + self._creating < self.config.pool.max_live_instances

    def _retire_error_locked(self) -> BaseException | None:
        """Return a retirement failure that prevents further capacity."""
        for state in self._states:
            if state.retire_error is not None:
                return state.retire_error
        return None


def _validate_config(config: KGBSessionConfig) -> None:
    if config.object_store.driver != config.sandbox_provider.data_transfer.driver:
        raise ValueError(
            f"KGB trace object store driver {config.object_store.driver!r} does not match "
            "its sandbox provider data-transfer driver "
            f"{config.sandbox_provider.data_transfer.driver!r}"
        )
    if config.pool.max_live_instances < 1:
        raise ValueError("KGB max_live_instances must be >= 1")
    if config.pool.max_active_sessions_per_instance < 1:
        raise ValueError("KGB max_active_sessions_per_instance must be >= 1")
    if config.pool.max_sessions_per_instance_lifetime < 1:
        raise ValueError("KGB max_sessions_per_instance_lifetime must be >= 1")
    if config.pool.prefetch_instances < 0:
        raise ValueError("KGB prefetch_instances must be >= 0")
    if config.gateway.trace.export_policy not in {"none", "on_release", "on_retire"}:
        raise ValueError("KGB trace_export must be one of: none, on_release, on_retire")
    if config.pool.idle_timeout_seconds < 1:
        raise ValueError("KGB idle_timeout_seconds must be >= 1")


def _kgb_config(config: KGBSessionConfig) -> KGBConfig:
    return KGBConfig(
        gateway=config.gateway,
        agent_runtime=config.agent_runtime,
        object_store=config.object_store,
        sandbox_provider=config.sandbox_provider,
        pool=config.pool,
        judge_model_id=config.judge_model_id,
        agent_models=dict(config.agent_models),
        trace_prefix=config.trace_prefix,
    )


def _service_record(
    *,
    context: TaskContext,
    agent_name: str,
    task_id: str,
) -> KGBServiceRecord:
    session_id = new_id("kgb")
    api_key_name = _make_kgb_key_name(
        agent_name,
        task_id,
        role="worker",
        identity=_key_identity(context, session_id),
    )
    return KGBServiceRecord(
        session_id=session_id,
        agent_name=agent_name,
        task_id=task_id,
        api_key_name=api_key_name,
        trace_file=f"{api_key_name}.jsonl",
        context=context,
    )


def _key_identity(context: TaskContext, session_id: str) -> str:
    return f"{context.run_id}-{context.attempt_id}-{session_id}"


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pool_context() -> TaskContext:
    return TaskContext(
        run_id="gateway",
        task_id="gateway",
        version=1,
        attempt_id="gateway",
        agent_name="gateway",
        agent_spec_name="gateway",
    )


async def _await_cleanup(awaitable: Awaitable[Any]) -> Any:
    """Shield a lifecycle cleanup operation from caller cancellation.

    Args:
        awaitable: Cleanup or retirement operation to protect.

    Returns:
        The awaitable result when the caller remains active.

    Raises:
        asyncio.CancelledError: After arranging for cleanup to continue in the
            background when the caller is cancelled.

    """
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.add_done_callback(_log_background_cleanup_result)
        raise


def _log_background_cleanup_result(task: asyncio.Future[Any]) -> None:
    """Observe and log a shielded background cleanup result."""
    try:
        task.result()
    except asyncio.CancelledError:
        logger.warning("KGB background cleanup was cancelled")
    except Exception:
        logger.exception("KGB background cleanup failed")
