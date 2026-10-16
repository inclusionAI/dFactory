"""Sandbox provider protocol and launch specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from anexus.core.context import TaskContext
from anexus.sandbox.network import SandboxNetworkPolicy
from anexus.sandbox.resources import SandboxResourceSnapshot


@dataclass(frozen=True)
class SandboxLease:
    """Provider-neutral runtime handle for an allocated sandbox.

    A lease is the only sandbox handle passed outside the concrete provider. It
    intentionally does not contain provider-native SDK objects. Runtime code
    should pass the lease back to the sandbox provider for follow-up operations
    like exec, file transfer, port lookup, and delete.

    The lease is primarily valid within the provider/control-plane scope that
    created it. Durable cleanup across process restarts may require additional
    provider-specific metadata, such as an AKernel logical sandbox name.

    Attributes:
        sandbox_id: Provider-assigned sandbox identifier returned after create.
            The concrete provider uses this value to look up the live sandbox
            handle and runtime events use it for audit/debug context.
        provider: Name of the sandbox provider that owns this lease. Routers use
            this value to dispatch subsequent operations to the correct provider.
    """

    sandbox_id: str
    provider: str


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class FixtureMapping:
    """Single fixture source-to-target mapping for cluster-internal download."""

    source_uri: str
    target_path: str
    kind: Literal["file", "directory"] = "directory"


@dataclass(frozen=True)
class ArtifactMapping:
    """Single artifact source-to-target mapping for cluster-internal upload."""

    source_path: str
    target_uri: str
    kind: Literal["file", "directory"] = "file"


@dataclass(frozen=True)
class DownloadResult:
    """Result of a single fixture download operation."""

    source_uri: str
    target_path: str
    kind: str
    size_bytes: int = 0
    file_count: int = 0


@dataclass(frozen=True)
class UploadResult:
    """Result of a single artifact upload operation."""

    source_path: str
    target_uri: str
    kind: str
    size_bytes: int = 0
    file_count: int = 0


@dataclass(frozen=True)
class SandboxProviderLimits:
    """Scheduler-visible resource limits exposed by one sandbox provider."""

    provider_name: str = "sandbox"
    active_sandboxes: int | None = None

    def __post_init__(self) -> None:
        provider_name = self.provider_name.strip()
        if not provider_name:
            raise ValueError("provider_name must be a non-empty string")
        if self.active_sandboxes is not None and self.active_sandboxes < 1:
            raise ValueError("active_sandboxes must be >= 1")
        object.__setattr__(self, "provider_name", provider_name)


@dataclass(frozen=True)
class SandboxLaunchSpec:
    """Provider-neutral fields required to allocate one sandbox.

    `SandboxLaunchSpec` is the provider-facing create payload. Runtime-only
    behavior such as task payload resolution, agent command execution,
    verification, sidecar health checks, and kubeconfig injection must be
    resolved before this object is passed to `SandboxProvider.create()`.

    The `command` field is intentionally launch-scoped. Providers may run it
    after the sandbox is allocated and file overrides are applied, typically as
    a background bootstrap process. It is similar to an image entrypoint in
    lifecycle intent, but it is not Docker/OCI `ENTRYPOINT` metadata and it is
    not the task runner command.

    Attributes:
        image: Container image used to allocate the sandbox.
        provider: Sandbox provider name that should handle this launch.
        port: Primary service port exposed by the sandbox. A value of `0`
            means no primary port is requested.
        cpu_request: CPU request passed to the provider scheduler.
        cpu_limit: CPU limit passed to the provider scheduler.
        memory_request_mb: Memory request in MiB.
        memory_limit_mb: Memory limit in MiB.
        command: Optional launch/bootstrap command. The AKernel provider starts
            it after sandbox creation and file overrides, before returning the
            lease. Agent task execution uses runtime `task_runner` instead.
        ports: Additional/explicit ports to forward. When set, providers should
            prefer this list over `port` for port-forwarding configuration.
        name: Stable logical sandbox name within the attempt, such as `main`,
            sidecar name, or `verify`.
        env: Environment variables injected into the sandbox at launch.
        file_overrides: Host-to-sandbox file copy declarations applied before
            `command` is started.
        network_policy: Provider-neutral network policy requested for this
            sandbox.
        extra_config: Provider-specific launch options that do not belong in
            the common protocol surface.
    """

    image: str
    provider: str
    port: int
    cpu_request: int
    cpu_limit: int
    memory_request_mb: int
    memory_limit_mb: int
    command: str | None = None
    ports: list[int] = field(default_factory=list)
    name: str = "main"
    env: dict[str, str] = field(default_factory=dict)
    file_overrides: list[dict[str, str]] = field(default_factory=list)
    network_policy: SandboxNetworkPolicy = field(default_factory=SandboxNetworkPolicy)
    extra_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "image": self.image,
            "provider": self.provider,
            "port": self.port,
            "cpu_request": self.cpu_request,
            "cpu_limit": self.cpu_limit,
            "memory_request_mb": self.memory_request_mb,
            "memory_limit_mb": self.memory_limit_mb,
        }
        if self.command:
            data["command"] = self.command
        if self.ports:
            data["ports"] = list(self.ports)
        if self.env:
            data["env"] = dict(self.env)
        if self.file_overrides:
            data["file_overrides"] = [dict(item) for item in self.file_overrides]
        if not self.network_policy.is_public:
            data["network_policy"] = self.network_policy.to_dict()
        if self.extra_config:
            data["extra_config"] = dict(self.extra_config)
        return data


class SandboxProvider(Protocol):
    """Sandbox control-plane contract consumed by runtime and API services.

    All methods are async. Cancellation propagates via asyncio.CancelledError.
    No cancel_event parameter — cancellation is handled by the event loop.
    """

    provider: str

    def control_plane_env(self, provider_name: str = "") -> dict[str, str]:
        """Return provider-specific control-plane env for nested sandboxes.

        Runtime code treats this as an opaque provider hook. Concrete providers
        own the actual environment variable names and lookup rules.
        """
        ...

    def limits(self) -> SandboxProviderLimits: ...

    def close(self) -> None:
        """Release provider-owned resources."""
        ...

    async def create(
        self,
        context: TaskContext,
        spec: SandboxLaunchSpec | None = None,
    ) -> SandboxLease: ...

    async def exec(
        self,
        lease: SandboxLease,
        command: str,
        *,
        stdin: str = "",
        timeout: int | float | None = None,
    ) -> ExecResult: ...

    async def put_file(self, lease: SandboxLease, source_path: str | Path, target_path: str) -> None: ...

    async def get_file(self, lease: SandboxLease, source_path: str, target_path: str | Path) -> None: ...

    async def port_url(self, lease: SandboxLease, port: int, *, internal: bool = False) -> str: ...

    async def delete(self, lease: SandboxLease) -> None: ...

    async def resource_snapshot(self) -> SandboxResourceSnapshot: ...

    # ──── Cluster-internal data operations ──────────────────────────

    async def download_fixtures(
        self,
        lease: SandboxLease,
        fixtures: list[FixtureMapping],
    ) -> list[DownloadResult]:
        """Download fixtures from OSS into sandbox. Cluster-internal operation.

        Processes each mapping sequentially. Raises on first failure;
        completed items are not rolled back (idempotent, retry-safe).
        """
        ...

    async def upload_artifacts(
        self,
        lease: SandboxLease,
        artifacts: list[ArtifactMapping],
    ) -> list[UploadResult]:
        """Upload artifacts from sandbox to OSS. Cluster-internal operation.

        Processes each mapping sequentially. Raises on first failure;
        completed items are not rolled back (idempotent, retry-safe).
        """
        ...

    async def copy_between(
        self,
        source_lease: SandboxLease,
        source_path: str,
        target_lease: SandboxLease,
        target_path: str,
        *,
        kind: Literal["file", "directory"] = "file",
    ) -> None:
        """Copy files between two sandboxes (cluster-internal).

        Used for isolated verification: copy main sandbox artifact
        to verifier sandbox.
        """
        ...
