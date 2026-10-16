"""Execute the run command and assemble its runtime providers."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import threading
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

from anexus.api.cli_context import (
    apply_workspace_environment,
    resolve_plan,
    restore_environment,
    workspace_from_input,
)
from anexus.api.command_inputs import RunCommandInput
from anexus.api.preflight import preflight_run_plan
from anexus.api.providers import (
    RuntimeProviderConfig,
    RuntimeProviderFactory,
    sandbox_provider_names_from_plan,
)
from anexus.config import RunPlan
from anexus.core.context import TaskContext
from anexus.events import InMemoryEventBus
from anexus.gateway.activity import GatewayActivityMonitor
from anexus.gateway.session import GatewaySessionProvider, KGBSessionConfig
from anexus.logs import run_file_logging
from anexus.runtime import (
    AttemptRunner,
    LocalRunStore,
    PreparedRun,
    RunController,
    RunLayout,
    RunResult,
    trace_storage_identity,
)
from anexus.runtime.store import record_event_safely
from anexus.ui import RunConsoleReporter

_GLOG_LOG_DIR_ENV = "GLOG_log_dir"
_GLOG_LOG_TO_STDERR_ENV = "GLOG_logtostderr"
_GRPC_VERBOSITY_ENV = "GRPC_VERBOSITY"


def run_command(command: RunCommandInput) -> int:
    """Validate and execute one typed run request."""

    saved_env = apply_workspace_environment(command)
    try:
        return _run_command_inner(command)
    finally:
        restore_environment(saved_env)


def _run_command_inner(command: RunCommandInput) -> int:
    plan = resolve_plan(command)
    validation = asyncio.run(preflight_run_plan(plan, check_object_fixtures=False))
    if validation.has_errors:
        messages = "; ".join(diagnostic.message for diagnostic in validation.diagnostics if diagnostic.level == "error")
        raise ValueError(f"run plan validation failed: {messages}")
    output_root = workspace_from_input(command).cache_dir
    store = LocalRunStore(output_root)
    kgb_session_configs = _create_kgb_session_configs(plan)
    bus = InMemoryEventBus(strict_sinks=False)
    trace_info: dict[str, Any] = {}

    with store.active_run(plan.run_id):
        resume = bool(command.run_id and store.run_dir(plan.run_id).exists())
        controller = RunController(
            bus=bus,
            run_store=store,
            resume_completed=resume,
            trace_storage=_trace_storage_from_kgb_configs(kgb_session_configs),
        )
        prepared = controller.prepare(plan)
        run_layout = RunLayout.open_existing(store.run_dir(plan.run_id))
        run_layout.ensure_dirs()
        run_layout.update_latest_link()
        native_output = _RunNativeOutput(run_layout.run_dir)
        native_output.prepare()
        application_log_path = run_layout.run_dir / "anexus.log"
        reporter_stream = native_output.open_original_stderr_stream()
        provider_factory: RuntimeProviderFactory | None = None
        gateway_sessions: GatewaySessionProvider | None = None
        activity_monitor: GatewayActivityMonitor | None = None
        reporter: RunConsoleReporter | None = None
        resource_provider: object | None = None
        runner: AttemptRunner | None = None
        try:
            with run_file_logging(application_log_path), native_output.redirect_stderr():
                if prepared.schedule.requests:
                    provider_factory = _runtime_provider_factory(
                        plan=plan,
                        kgb_session_configs=kgb_session_configs,
                    )
                    provider_names = sandbox_provider_names_from_plan(plan)
                    provider_names.extend(config.sandbox_provider.name for config in kgb_session_configs.values())
                    resource_provider = provider_factory.create_sandbox_provider(provider_names)
                    gateway_sessions = provider_factory.create_gateway_sessions()

                    def publish_activity(event):
                        bus.publish(event)
                        record_event_safely(store, event, bus.publish)

                    activity_monitor = GatewayActivityMonitor(
                        gateway_sessions,
                        publish_activity,
                    )
                    runner = provider_factory.create_executor(
                        output_root=output_root,
                        bus=bus,
                    )

                reporter = _run_console_reporter(
                    gateway_sessions,
                    stream=reporter_stream,
                    output_dir=run_layout.run_dir,
                    native_stderr_path=native_output.stderr_path,
                    application_log_path=application_log_path,
                    activity_snapshots=(activity_monitor.snapshots if activity_monitor is not None else None),
                )
                if reporter is not None:
                    bus.sinks.append(reporter)
                    reporter.start()

                viewer_urls = {
                    agent_name: url
                    for agent_name, config in kgb_session_configs.items()
                    if (url := config.trace_viewer_url())
                }
                if viewer_urls:
                    trace_info = {"trace_viewer_urls": viewer_urls}
                    for agent_name, viewer_url in viewer_urls.items():
                        logging.getLogger(__name__).info("Trace Viewer [%s]: %s", agent_name, viewer_url)
                        controller.log(
                            "run.trace_viewer",
                            plan,
                            agent_name=agent_name,
                            trace_viewer_url=viewer_url,
                        )

                try:
                    with _quiet_run_console_noise():
                        result = _run_async_with_sigint_cleanup(
                            _run_controller(
                                plan=plan,
                                context=_gateway_service_context(plan),
                                controller=controller,
                                prepared=prepared,
                                runner=runner,
                                resource_provider=resource_provider,
                                gateway_sessions=gateway_sessions,
                                activity_monitor=activity_monitor,
                            )
                        )
                except KeyboardInterrupt:
                    logging.getLogger(__name__).info("run interrupted after cleanup")
                    raise
                except BaseException:
                    logging.getLogger(__name__).exception("run failed")
                    raise
                finally:
                    if reporter is not None:
                        reporter.stop()
        finally:
            try:
                if provider_factory is not None:
                    provider_factory.close()
            finally:
                native_output.close()

        output_payload: dict[str, Any] = {
            "run_id": result.run_id,
            "agent_names": list(result.agent_names),
            "status": result.status,
            "record_count": len(result.records),
            "status_counts": result.status_counts,
            "evaluation_counts": result.evaluation_counts,
            "total_slot_count": result.total_slot_count,
            "resumed_slot_count": result.resumed_slot_count,
            "scheduled_slot_count": result.scheduled_slot_count,
            "executed_attempt_count": result.executed_attempt_count,
            "physical_attempt_count": result.physical_attempt_count,
            "output_root": str(output_root),
        }
        if trace_info:
            output_payload.update(trace_info)
        print(json.dumps(output_payload, ensure_ascii=False, sort_keys=True))
        return 0


async def _run_controller(
    *,
    plan: RunPlan,
    context: TaskContext,
    controller: RunController,
    prepared: PreparedRun,
    runner: AttemptRunner | None,
    resource_provider: object | None,
    gateway_sessions: GatewaySessionProvider | None,
    activity_monitor: GatewayActivityMonitor | None,
) -> RunResult:
    """Run gateway and task lifecycles on one event loop.

    Args:
        plan: Fully resolved run plan.
        context: Run-scoped gateway context.
        controller: Run controller.
        runner: Attempt executor consumed by the controller.
        resource_provider: Sandbox provider used for resource snapshots.
        gateway_sessions: Run-scoped gateway service.

    Returns:
        The aggregate run result.

    """
    primary_error: BaseException | None = None
    monitor_task: asyncio.Task[None] | None = None
    if not prepared.schedule.requests:
        result = await controller.run_async(
            plan,
            runner=None,
            prepared=prepared,
        )
        return result
    if gateway_sessions is None or activity_monitor is None or runner is None:
        raise RuntimeError("pending run slots require initialized runtime providers")
    try:
        await gateway_sessions.start(context)
        monitor_task = asyncio.create_task(
            activity_monitor.run(),
            name="anexus-gateway-activity",
        )
        result = await controller.run_async(
            plan,
            runner=runner,
            sandbox_provider=resource_provider,
            prepared=prepared,
        )
        return result
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        activity_monitor.stop()
        if monitor_task is not None:
            try:
                await monitor_task
            except Exception:
                logging.getLogger(__name__).exception("gateway activity monitor failed")
        try:
            await gateway_sessions.stop(graceful=primary_error is None or not _is_interruption(primary_error))
        except BaseException:
            if primary_error is None:
                raise
            logging.getLogger(__name__).exception("gateway stop also failed")


def _run_async_with_sigint_cleanup(awaitable):
    """Run one CLI coroutine while reserving the first SIGINT for cleanup."""

    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        main_task = loop.create_task(awaitable)

        async def wait_for_main():
            return await main_task

        if threading.current_thread() is not threading.main_thread():
            return runner.run(wait_for_main())

        previous_handler = signal.getsignal(signal.SIGINT)
        interrupt_count = 0

        def handle_sigint(_signum, _frame) -> None:
            nonlocal interrupt_count
            interrupt_count += 1
            if interrupt_count == 1:
                os.write(
                    2,
                    b"\nInterrupt received; cleaning active resources. "
                    b"Press Ctrl-C again to exit immediately; resources may remain.\n",
                )
                loop.call_soon_threadsafe(main_task.cancel)
                return
            os.write(
                2,
                b"\nSecond interrupt received; exiting immediately. Active resources may remain.\n",
            )
            os._exit(130)

        signal.signal(signal.SIGINT, handle_sigint)
        try:
            try:
                return runner.run(wait_for_main())
            except asyncio.CancelledError:
                if interrupt_count:
                    raise KeyboardInterrupt from None
                raise
        finally:
            signal.signal(signal.SIGINT, previous_handler)


def _gateway_service_context(plan: RunPlan) -> TaskContext:
    """Build a short naming context for the run-scoped KGB service."""
    return TaskContext(
        run_id=plan.run_id,
        task_id="service",
        version=1,
        attempt_id=plan.run_id[:8],
        agent_name="run",
        agent_spec_name="run",
    )


def _is_interruption(exc: BaseException) -> bool:
    """Return whether an exception represents process-level cancellation."""

    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_interruption(child) for child in exc.exceptions)
    return False


def _run_console_reporter(
    gateway_sessions: GatewaySessionProvider | None,
    *,
    stream: TextIO | None = None,
    output_dir: Path | None = None,
    native_stderr_path: Path | None = None,
    application_log_path: Path | None = None,
    activity_snapshots=None,
) -> RunConsoleReporter | None:
    mode = _run_progress_mode()
    if mode == "off":
        return None
    return RunConsoleReporter(
        stream=stream,
        mode=mode,
        status_interval=_run_status_interval(),
        kgb_log_counts=lambda task_ids: _kgb_log_counts_by_task(gateway_sessions, task_ids),
        kgb_total_logs=lambda: _kgb_total_logs(gateway_sessions),
        output_dir=output_dir,
        native_stderr_path=native_stderr_path,
        application_log_path=application_log_path,
        activity_snapshots=activity_snapshots,
    )


def _run_progress_mode() -> str:
    raw = os.environ.get("ANEXUS_RUN_PROGRESS", "compact").strip().lower()
    if raw in {"", "1", "true", "yes", "compact"}:
        return "compact"
    if raw in {"verbose", "debug"}:
        return "verbose"
    if raw in {"0", "false", "no", "none", "off"}:
        return "off"
    return "compact"


def _run_status_interval() -> float:
    raw = os.environ.get("ANEXUS_RUN_STATUS_INTERVAL", "").strip()
    if not raw:
        return 10.0
    try:
        interval = float(raw)
    except ValueError:
        return 10.0
    return interval if interval > 0 else 10.0


def _kgb_log_counts_by_task(
    gateway_sessions: GatewaySessionProvider | None,
    task_ids: Iterable[str],
) -> Mapping[str, int]:
    if gateway_sessions is None:
        return {}
    getter = getattr(gateway_sessions, "kgb_log_counts_by_task", None)
    if not callable(getter):
        return {}
    return getter(task_ids)


def _kgb_total_logs(gateway_sessions: GatewaySessionProvider | None) -> int:
    if gateway_sessions is None:
        return 0
    getter = getattr(gateway_sessions, "kgb_log_count", None)
    if not callable(getter):
        return 0
    try:
        return int(getter())
    except (TypeError, ValueError):
        return 0


def _trace_storage_from_kgb_configs(
    configs: Mapping[str, KGBSessionConfig],
) -> dict[str, object]:
    return {
        agent_name: trace_storage_identity(
            endpoint=config.object_store.endpoint,
            bucket=config.object_store.bucket,
            prefix=f"{config.trace_prefix}/traces",
        )
        for agent_name, config in configs.items()
    }


@contextlib.contextmanager
def _quiet_run_console_noise():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*doesn't match a supported version.*",
        )
        yield


class _RunNativeOutput:
    """Run-scoped native stderr and glog destinations."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.native_dir = run_dir / "native"
        self.stderr_path = self.native_dir / "stderr.log"
        self.glog_dir = self.native_dir / "glog"
        self._original_stderr_stream: TextIO | None = None
        self._owns_original_stream = False

    def prepare(self) -> None:
        self.glog_dir.mkdir(parents=True, exist_ok=True)
        self.stderr_path.touch(exist_ok=True)

    def open_original_stderr_stream(self) -> TextIO:
        if self._original_stderr_stream is not None:
            return self._original_stderr_stream
        try:
            fd = os.dup(2)
        except OSError:
            self._original_stderr_stream = sys.stderr
            self._owns_original_stream = False
            return self._original_stderr_stream
        self._original_stderr_stream = os.fdopen(
            fd,
            "w",
            buffering=1,
            encoding=getattr(sys.stderr, "encoding", None) or "utf-8",
            errors=getattr(sys.stderr, "errors", None) or "replace",
        )
        self._owns_original_stream = True
        return self._original_stderr_stream

    @contextlib.contextmanager
    def redirect_stderr(self):
        self.prepare()
        old_glog_dir = os.environ.get(_GLOG_LOG_DIR_ENV)
        old_glog_to_stderr = os.environ.get(_GLOG_LOG_TO_STDERR_ENV)
        old_grpc_verbosity = os.environ.get(_GRPC_VERBOSITY_ENV)
        os.environ.setdefault(_GLOG_LOG_TO_STDERR_ENV, "0")
        os.environ.setdefault(_GRPC_VERBOSITY_ENV, "ERROR")
        os.environ[_GLOG_LOG_DIR_ENV] = str(self.glog_dir)
        saved_stderr_fd: int | None = None
        log_stream: TextIO | None = None
        try:
            with contextlib.suppress(Exception):
                sys.stderr.flush()
            saved_stderr_fd = os.dup(2)
            log_stream = self.stderr_path.open(
                "a",
                buffering=1,
                encoding="utf-8",
                errors="replace",
            )
            os.dup2(log_stream.fileno(), 2)
            yield
        finally:
            with contextlib.suppress(Exception):
                sys.stderr.flush()
            if saved_stderr_fd is not None:
                with contextlib.suppress(OSError):
                    os.dup2(saved_stderr_fd, 2)
                with contextlib.suppress(OSError):
                    os.close(saved_stderr_fd)
            if log_stream is not None:
                log_stream.close()
            if old_glog_dir is None:
                os.environ.pop(_GLOG_LOG_DIR_ENV, None)
            else:
                os.environ[_GLOG_LOG_DIR_ENV] = old_glog_dir
            _restore_env_value(_GLOG_LOG_TO_STDERR_ENV, old_glog_to_stderr)
            _restore_env_value(_GRPC_VERBOSITY_ENV, old_grpc_verbosity)

    def close(self) -> None:
        if self._original_stderr_stream is not None and self._owns_original_stream:
            with contextlib.suppress(Exception):
                self._original_stderr_stream.close()
        self._original_stderr_stream = None
        self._owns_original_stream = False


def _restore_env_value(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _create_kgb_session_configs(
    plan: RunPlan,
) -> dict[str, KGBSessionConfig]:
    """Create one KGB gateway pool configuration per Agent."""
    return {agent_name: _create_kgb_session_config(plan, agent_name) for agent_name in plan.agent_names}


def _create_kgb_session_config(
    plan: RunPlan,
    agent_name: str,
) -> KGBSessionConfig:
    agent = plan.agent(agent_name)
    if agent.gateway.driver != "kgb":
        raise ValueError(f"unsupported gateway driver: {agent.gateway.driver}")
    if not agent.sandbox.task_runner:
        raise ValueError(f"agent {agent_name!r} does not define sandbox.task_runner")
    gateway = agent.gateway
    object_store = plan.object_stores[gateway.trace.object_store]
    sandbox_provider = plan.sandbox_providers[gateway.sandbox_provider]
    trace_prefix = "/".join(
        part.strip("/") for part in (object_store.prefix, "runs", plan.run_id, agent_name) if part.strip("/")
    )

    return KGBSessionConfig(
        gateway=gateway,
        agent_runtime=agent.agent_runtime,
        object_store=object_store,
        sandbox_provider=sandbox_provider,
        pool=gateway.pool,
        judge_model_id=agent.judge_model_id,
        agent_models=dict(agent.agent_models),
        trace_prefix=trace_prefix,
    )


def _runtime_provider_factory(
    *,
    plan: RunPlan,
    kgb_session_configs: dict[str, KGBSessionConfig] | None = None,
) -> RuntimeProviderFactory:
    return RuntimeProviderFactory(
        RuntimeProviderConfig(
            sandbox_providers=plan.sandbox_providers,
            object_stores=plan.object_stores,
            default_sandbox_provider=next(iter(plan.sandbox_providers)),
            kgb_session_configs=dict(kgb_session_configs or {}),
            agent_spec_names={name: agent.agent_spec.name for name, agent in plan.agents.items()},
        )
    )
