"""Sandbox provider registry."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import Parameter, signature
from pathlib import Path
from typing import Literal

from anexus.core.context import TaskContext
from anexus.sandbox.provider import (
    ArtifactMapping,
    DownloadResult,
    ExecResult,
    FixtureMapping,
    SandboxLaunchSpec,
    SandboxLease,
    SandboxProvider,
    SandboxProviderLimits,
    UploadResult,
)
from anexus.sandbox.resources import SandboxResourceSnapshot

ProviderFactory = Callable[[], SandboxProvider]


@dataclass
class SandboxProviderRegistry:
    """Map provider names to provider factories."""

    _factories: dict[str, ProviderFactory] = field(default_factory=dict)

    def register(self, name: str, factory: ProviderFactory) -> None:
        if not name:
            raise ValueError("sandbox provider name is required")
        if name in self._factories:
            raise ValueError(f"sandbox provider already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str) -> SandboxProvider:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"unknown sandbox provider {name!r}; known providers: {known}") from exc
        return factory()

    def names(self) -> list[str]:
        return sorted(self._factories)


@dataclass
class SandboxProviderRouter:
    """Route sandbox calls to provider instances by provider name.

    Runtime sandbox lifecycle calls are async. Synchronous CLI boundaries should
    enter the async control plane with `asyncio.run()` instead of routing
    provider calls through sync compatibility wrappers.
    """

    registry: SandboxProviderRegistry
    default_provider: str
    _providers: dict[str, SandboxProvider] = field(default_factory=dict)
    _exec_accepts_timeout: dict[str, bool] = field(default_factory=dict)

    @property
    def provider(self) -> str:
        return self.default_provider

    def control_plane_env(self, provider_name: str = "") -> dict[str, str]:
        provider_name = provider_name or self.default_provider
        provider = self._provider_for(provider_name)
        control_plane_env = getattr(provider, "control_plane_env", None)
        if not callable(control_plane_env):
            return {}
        return dict(control_plane_env(provider_name))

    def limits(self) -> SandboxProviderLimits:
        return _provider_limits(
            self._provider_for(self.default_provider),
            self.default_provider,
        )

    async def create(
        self,
        context: TaskContext,
        spec: SandboxLaunchSpec | None = None,
    ) -> SandboxLease:
        provider_name = spec.provider if spec is not None and spec.provider else self.default_provider
        provider = self._provider_for(provider_name)
        return await provider.create(context, spec)

    async def exec(
        self,
        lease: SandboxLease,
        command: str,
        *,
        stdin: str = "",
        timeout: int | float | None = None,
    ) -> ExecResult:
        provider_name = lease.provider
        exec_call = self._provider_for(provider_name).exec
        kwargs: dict[str, object] = {"stdin": stdin}
        if timeout is not None:
            accepts_timeout = self._exec_accepts_timeout.get(provider_name)
            if accepts_timeout is None:
                accepts_timeout = _call_accepts_keyword(exec_call, "timeout")
                self._exec_accepts_timeout[provider_name] = accepts_timeout
            if accepts_timeout:
                kwargs["timeout"] = timeout
        return await exec_call(lease, command, **kwargs)

    async def put_file(self, lease: SandboxLease, source_path: str | Path, target_path: str) -> None:
        await self._provider_for(lease.provider).put_file(lease, source_path, target_path)

    async def get_file(self, lease: SandboxLease, source_path: str, target_path: str | Path) -> None:
        await self._provider_for(lease.provider).get_file(lease, source_path, target_path)

    async def port_url(self, lease: SandboxLease, port: int, *, internal: bool = False) -> str:
        return await self._provider_for(lease.provider).port_url(lease, port, internal=internal)

    async def delete(self, lease: SandboxLease) -> None:
        await self._provider_for(lease.provider).delete(lease)

    async def download_fixtures(
        self,
        lease: SandboxLease,
        fixtures: list[FixtureMapping],
    ) -> list[DownloadResult]:
        provider = self._provider_for(lease.provider)
        return await provider.download_fixtures(lease, fixtures)

    async def upload_artifacts(
        self,
        lease: SandboxLease,
        artifacts: list[ArtifactMapping],
    ) -> list[UploadResult]:
        provider = self._provider_for(lease.provider)
        return await provider.upload_artifacts(lease, artifacts)

    async def copy_between(
        self,
        source_lease: SandboxLease,
        source_path: str,
        target_lease: SandboxLease,
        target_path: str,
        *,
        kind: Literal["file", "directory"] = "file",
    ) -> None:
        provider = self._provider_for(source_lease.provider)
        await provider.copy_between(source_lease, source_path, target_lease, target_path, kind=kind)

    async def resource_snapshot(self) -> SandboxResourceSnapshot:
        return await self._provider_for(self.default_provider).resource_snapshot()

    async def resource_snapshots(self) -> tuple[SandboxResourceSnapshot, ...]:
        providers = [self._provider_for(provider_name) for provider_name in self.registry.names()]
        snapshots = await asyncio.gather(*(provider.resource_snapshot() for provider in providers))
        return tuple(snapshots)

    def close(self) -> None:
        """Close provider instances created by this router."""

        for provider in self._providers.values():
            close = getattr(provider, "close", None)
            if callable(close):
                close()
        self._providers.clear()
        self._exec_accepts_timeout.clear()

    def _provider_for(self, name: str) -> SandboxProvider:
        if name not in self._providers:
            self._providers[name] = self.registry.create(name)
        return self._providers[name]


def _provider_limits(
    provider: SandboxProvider,
    provider_name: str,
) -> SandboxProviderLimits:
    limits = getattr(provider, "limits", None)
    if callable(limits):
        return limits()
    return SandboxProviderLimits(provider_name=provider_name)


def _call_accepts_keyword(call: Callable[..., object], keyword: str) -> bool:
    try:
        parameters = signature(call).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.kind == Parameter.VAR_KEYWORD or parameter.name == keyword for parameter in parameters)
