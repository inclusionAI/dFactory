"""Gateway session protocols and provider implementations.

The runtime only needs a small gateway-facing surface: checkout one session for
an attempt, inject its environment into the agent sandbox, and release the
session during cleanup. KGB pooling, trace export, and sandbox lifecycle are
gateway-internal details.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from anexus.config.catalog import (
    AgentRuntimeSpec,
    GatewayPoolSpec,
    GatewaySpec,
    ObjectStoreSpec,
    SandboxProviderSpec,
)
from anexus.core.context import TaskContext
from anexus.core.errors import AnexusError, ErrorKind
from anexus.core.execution import GatewayPlan
from anexus.sandbox.provider import SandboxProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GatewayActivitySnapshot:
    """Latest request activity observed for one gateway session."""

    context: TaskContext
    gateway_session_id: str
    api_key_name: str = ""
    api_key_id: int | None = None
    request_count: int = 0
    last_request_time: str = ""
    response_status: int | None = None
    first_byte_delay_ms: int | None = None
    total_time_ms: int | None = None
    retry_count: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    available: bool = True
    error_type: str = ""
    error: str = ""
    observed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def event_context(self) -> dict[str, object]:
        context = self.context.to_fields()
        context["gateway_session_id"] = self.gateway_session_id
        return context

    def event_payload(self, *, sample: str, request_delta: int = 0) -> dict[str, object]:
        payload: dict[str, object] = {
            "sample": sample,
            "available": self.available,
            "request_count": self.request_count,
            "request_delta": request_delta,
            "last_request_time": self.last_request_time,
            "retry_count": self.retry_count,
            "observed_at": self.observed_at,
        }
        optional = {
            "api_key_id": self.api_key_id,
            "api_key_name": self.api_key_name,
            "response_status": self.response_status,
            "first_byte_delay_ms": self.first_byte_delay_ms,
            "total_time_ms": self.total_time_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error_type": self.error_type,
            "error": self.error,
        }
        payload.update({key: value for key, value in optional.items() if value not in (None, "")})
        return payload


class GatewaySession(Protocol):
    """Attempt-scoped gateway lease returned to the runtime.

    A gateway session is not a sandbox lease. It is a gateway service lease that
    knows how to produce agent LLM environment variables and how to release one
    unit of gateway service capacity after the attempt finishes.
    """

    session_id: str

    def agent_env(self) -> dict[str, str]:
        """Return environment variables injected into the agent sandbox.

        Returns:
            A dictionary of environment variables. No-gateway sessions return an
            empty dictionary.

        """
        ...

    def trace_refs(self) -> list[dict[str, str]]:
        """Return trace references associated with this attempt.

        Returns:
            A list of JSON-serializable trace references. The references may
            point to traces that are exported later, depending on the provider's
            trace export policy.

        """
        ...

    async def activity_snapshot(self) -> GatewayActivitySnapshot | None:
        """Return the latest request activity for this session, when available."""
        ...

    async def release(
        self,
        *,
        export_traces: bool = False,
        graceful: bool = True,
    ) -> list[str]:
        """Release this attempt's gateway service lease.

        Args:
            export_traces: Whether to request trace export for this specific
                attempt during release.
            graceful: Whether optional trace export should run before release.

        Returns:
            Trace file names or URIs exported during release.

        """
        ...


class GatewaySessionProvider(Protocol):
    """Run-scoped gateway session provider used by attempt executors."""

    async def start(self, context: TaskContext) -> None:
        """Start provider-owned background resources for a run.

        Args:
            context: Run or session context used for diagnostics.

        """
        ...

    async def checkout(
        self,
        context: TaskContext,
        spec: GatewayPlan,
    ) -> GatewaySession:
        """Checkout an attempt-scoped gateway session.

        Args:
            context: Attempt context requesting a session.
            spec: Statically compiled route and model checkout intent.

        Returns:
            A gateway session for one attempt.

        """
        ...

    async def stop(self, *, graceful: bool = True) -> None:
        """Stop provider resources, optionally flushing observability data."""
        ...

    async def activity_snapshots(self) -> list[GatewayActivitySnapshot]:
        """Return request activity for all active gateway sessions."""
        ...


class NoopGatewaySession:
    """Gateway session used when a run does not need a gateway."""

    def __init__(self) -> None:
        """Initialize a no-op attempt gateway session."""
        self.session_id = ""
        self._released = False

    def agent_env(self) -> dict[str, str]:
        """Return an empty agent environment."""
        return {}

    def trace_refs(self) -> list[dict[str, str]]:
        """Return no trace references."""
        return []

    async def activity_snapshot(self) -> GatewayActivitySnapshot | None:
        """Return no activity for a no-op session."""
        return None

    async def release(
        self,
        *,
        export_traces: bool = False,
        graceful: bool = True,
    ) -> list[str]:
        """Mark the no-op session released.

        Args:
            export_traces: Ignored for no-op sessions.
            graceful: Ignored for no-op sessions.

        Returns:
            An empty list.

        """
        self._released = True
        return []


class NoopGatewaySessionProvider:
    """Run-scoped gateway provider for runs without gateway routing."""

    async def start(self, context: TaskContext) -> None:
        """Start the provider.

        Args:
            context: Ignored.

        """
        return None

    async def checkout(
        self,
        context: TaskContext,
        spec: GatewayPlan,
    ) -> GatewaySession:
        """Return a no-op gateway session.

        Args:
            context: Ignored.
            spec: Ignored.

        Returns:
            A `NoopGatewaySession`.

        """
        return NoopGatewaySession()

    async def stop(self, *, graceful: bool = True) -> None:
        """Stop the provider."""
        return None

    async def activity_snapshots(self) -> list[GatewayActivitySnapshot]:
        """Return no activity for a no-op provider."""
        return []


class GatewaySessionProviderRouter:
    """Route attempt gateway sessions to the provider for their Agent."""

    def __init__(
        self,
        providers: Mapping[str, GatewaySessionProvider],
        *,
        agent_spec_names: Mapping[str, str] | None = None,
    ) -> None:
        if not providers:
            raise ValueError("gateway session router requires at least one Agent provider")
        self._providers = dict(providers)
        self._agent_spec_names = dict(agent_spec_names or {})
        self._started: list[str] = []

    async def start(self, context: TaskContext) -> None:
        """Start every Agent-scoped provider, rolling back partial startup."""
        if self._started:
            raise RuntimeError("gateway session router is already started")
        try:
            for index, (agent_name, provider) in enumerate(self._providers.items()):
                service_context = TaskContext(
                    run_id=context.run_id,
                    task_id="service",
                    version=context.version,
                    attempt_id=f"{context.attempt_id}-{index}",
                    agent_name=agent_name,
                    agent_spec_name=self._agent_spec_names.get(agent_name, agent_name),
                )
                await provider.start(service_context)
                self._started.append(agent_name)
        except BaseException:
            await self.stop(graceful=False)
            raise

    async def checkout(self, context: TaskContext, spec: GatewayPlan) -> GatewaySession:
        """Checkout from the provider selected by ``context.agent_name``."""
        try:
            provider = self._providers[context.agent_name]
        except KeyError as exc:
            raise ValueError(f"no gateway session provider for agent {context.agent_name!r}") from exc
        return await provider.checkout(context, spec)

    async def stop(self, *, graceful: bool = True) -> None:
        """Stop all started providers in reverse startup order."""
        first_error: BaseException | None = None
        names = list(reversed(self._started))
        self._started.clear()
        for agent_name in names:
            try:
                await self._providers[agent_name].stop(graceful=graceful)
            except BaseException as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error

    async def activity_snapshots(self) -> list[GatewayActivitySnapshot]:
        """Return activity from all Agent-scoped providers."""
        snapshots: list[GatewayActivitySnapshot] = []
        for provider in self._providers.values():
            snapshots.extend(await provider.activity_snapshots())
        return snapshots

    def trace_viewer_urls(self) -> dict[str, str]:
        """Return non-empty trace viewers keyed by Agent name."""
        urls: dict[str, str] = {}
        for agent_name, provider in self._providers.items():
            getter = getattr(provider, "trace_viewer_url", None)
            if callable(getter) and (url := str(getter())):
                urls[agent_name] = url
        return urls

    def kgb_log_count(self) -> int:
        return sum(
            int(getter())
            for provider in self._providers.values()
            if callable(getter := getattr(provider, "kgb_log_count", None))
        )

    def kgb_log_counts_by_task(self, task_ids: Iterable[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        requested = list(task_ids)
        for provider in self._providers.values():
            getter = getattr(provider, "kgb_log_counts_by_task", None)
            if not callable(getter):
                continue
            for task_id, count in getter(requested).items():
                counts[task_id] = counts.get(task_id, 0) + int(count)
        return counts


@dataclass
class KGBSessionConfig:
    """Resolved catalog configuration for KGB-backed gateway sessions."""

    gateway: GatewaySpec
    agent_runtime: AgentRuntimeSpec
    object_store: ObjectStoreSpec
    sandbox_provider: SandboxProviderSpec
    pool: GatewayPoolSpec
    judge_model_id: str
    agent_models: dict[str, str]
    trace_prefix: str

    def trace_viewer_url(self) -> str:
        """Return the run-scoped viewer URL without starting a KGB sandbox."""
        if not self.gateway.trace.viewer_url:
            return ""
        viewer_config = json.dumps(
            {
                "endpoint": self.object_store.public_endpoint,
                "accessKeyId": self.object_store.access_key_id,
                "accessKeySecret": self.object_store.access_key_secret,
                "bucketName": self.object_store.bucket,
                "prefix": f"{self.trace_prefix}/traces",
            },
            separators=(",", ":"),
        )
        encoded = base64.b64encode(viewer_config.encode()).decode()
        return f"{self.gateway.trace.viewer_url.rstrip('/')}#/oss/{encoded}"


class KGBGatewaySessionProvider:
    """Gateway session provider backed by a KGB service pool."""

    def __init__(
        self,
        config: KGBSessionConfig,
        *,
        sandbox_provider: SandboxProvider,
    ) -> None:
        """Initialize the provider.

        Args:
            config: KGB session configuration.
            sandbox_provider: Run-scoped sandbox provider used to create KGB
                sandboxes.

        """
        from anexus.gateway.kgb.pool import KGBPool

        self._pool = KGBPool(config, sandbox_provider=sandbox_provider)

    async def start(self, context: TaskContext) -> None:
        """Start KGB prefetch for the run.

        Args:
            context: Run or session context used for KGB sandbox names and
                diagnostics.

        """
        await self._pool.start(context)

    async def checkout(
        self,
        context: TaskContext,
        spec: GatewayPlan,
    ) -> GatewaySession:
        """Checkout a KGB-backed gateway session.

        Args:
            context: Attempt context requesting a KGB session.
            spec: Statically compiled KGB route and model.

        Returns:
            A KGB-backed gateway session.

        """
        if spec.route != "kgb":
            raise ValueError(f"KGB gateway provider cannot serve route {spec.route!r}")
        return await self._pool.checkout(
            context,
            agent_name=context.agent_name,
            task_id=context.task_id,
        )

    async def stop(self, *, graceful: bool = True) -> None:
        """Retire all live KGB instances."""
        await self._pool.stop(graceful=graceful)

    def trace_viewer_url(self) -> str:
        """Return a trace viewer URL for this provider, when configured."""
        return self._pool.trace_viewer_url()

    def kgb_log_count(self) -> int:
        """Return the total KGB log count across live instances."""
        return self._pool.kgb_log_count()

    def kgb_log_counts_by_task(self, task_ids: Iterable[str]) -> dict[str, int]:
        """Return KGB log counts grouped by task.

        Args:
            task_ids: Task identifiers to query.

        Returns:
            Mapping from task id to KGB log count.

        """
        return self._pool.kgb_log_counts_by_task(task_ids)

    async def activity_snapshots(self) -> list[GatewayActivitySnapshot]:
        """Return request activity for all active KGB sessions."""
        return await self._pool.activity_snapshots()


def gateway_session_error(
    exc: BaseException,
    *,
    phase: str,
    context: TaskContext,
) -> AnexusError:
    """Wrap a gateway session failure in an `AnexusError`.

    Args:
        exc: Original exception.
        phase: Gateway phase that failed.
        context: Task context associated with the failure.

    Returns:
        A structured gateway error.

    """
    gateway_phase = f"gateway.kgb.{phase}"
    if isinstance(exc, AnexusError):
        if exc.component == "gateway.kgb" and exc.details.get("phase") == gateway_phase:
            return exc
        message = str(exc) or exc.__class__.__name__
        return AnexusError(
            f"KGB gateway {phase} failed: {message}",
            kind=exc.kind,
            component="gateway.kgb",
            retryable=exc.retryable,
            details={
                "phase": gateway_phase,
                "exception_type": exc.__class__.__name__,
                "original_kind": exc.kind.value,
                "original_component": exc.component,
                "original_message": message,
                "original_details": dict(exc.details),
                **context.to_fields(),
            },
        )
    message = str(exc) or exc.__class__.__name__
    return AnexusError(
        f"KGB gateway {phase} failed: {message}",
        kind=ErrorKind.INFRASTRUCTURE_ERROR,
        component="gateway.kgb",
        retryable=True,
        details={
            "phase": gateway_phase,
            "exception_type": exc.__class__.__name__,
            "original_message": message,
            **context.to_fields(),
        },
    )
