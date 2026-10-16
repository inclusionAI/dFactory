"""API-layer provider assembly for runtime execution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from anexus.config import ObjectStoreSpec, RunPlan, SandboxProviderSpec
from anexus.events import EventBus
from anexus.gateway.session import (
    GatewaySessionProvider,
    GatewaySessionProviderRouter,
    KGBGatewaySessionProvider,
    KGBSessionConfig,
    NoopGatewaySessionProvider,
)
from anexus.outputs import LocalOutputPublisher
from anexus.runtime import AsyncTaskExecutor, CleanupBacklog, LocalRunStore
from anexus.runtime.planning import (
    compile_evaluation_plan,
    compile_main_sandbox_plan,
    compile_sidecar_plans,
)
from anexus.sandbox import (
    SandboxLaunchSpec,
    SandboxProvider,
    SandboxProviderRegistry,
    SandboxProviderRouter,
)
from anexus.sandbox.akernel import AkernelSandboxProvider


@dataclass(frozen=True)
class SandboxProviderConfig:
    """Resolved sandbox resources chosen from the catalog."""

    providers: dict[str, SandboxProviderSpec]
    object_stores: dict[str, ObjectStoreSpec]
    default_provider: str

    def __post_init__(self) -> None:
        """Validate provider endpoint configuration."""
        if not self.default_provider:
            raise ValueError("sandbox provider name is required")
        if self.default_provider not in self.providers:
            raise ValueError(f"unknown default sandbox provider {self.default_provider!r}")


class SandboxProviderFactory:
    """Create sandbox providers from endpoint configuration."""

    def __init__(self, config: SandboxProviderConfig):
        """Initialize the factory from API-layer configuration."""
        self.config = config

    def create(self, provider_names: Iterable[str] = ()) -> SandboxProvider:
        """Create a provider router for the requested provider names."""
        registry = SandboxProviderRegistry()
        names = {self.config.default_provider, *(name for name in provider_names if name)}
        for name in names:
            try:
                spec = self.config.providers[name]
            except KeyError as exc:
                raise ValueError(f"unknown sandbox provider {name!r}") from exc
            registry.register(
                name,
                lambda spec=spec: _sandbox_provider_from_spec(
                    spec,
                    object_stores=self.config.object_stores,
                ),
            )
        return SandboxProviderRouter(
            registry=registry,
            default_provider=self.config.default_provider,
        )


@dataclass(frozen=True)
class RuntimeProviderConfig:
    """Endpoint and provider names chosen by the API layer for one run scope."""

    sandbox_providers: dict[str, SandboxProviderSpec]
    object_stores: dict[str, ObjectStoreSpec]
    default_sandbox_provider: str
    kgb_session_configs: dict[str, KGBSessionConfig] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    agent_spec_names: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate runtime provider configuration."""
        if not self.default_sandbox_provider:
            raise ValueError("sandbox provider name is required")


@dataclass(frozen=True)
class AgentSandboxLaunch:
    """One preflight launch associated with its owning Agent."""

    agent_name: str
    agent_spec_name: str
    launch: SandboxLaunchSpec


class RuntimeProviderFactory:
    """Create gateway, sandbox, engine, and executor objects from API config."""

    def __init__(self, config: RuntimeProviderConfig):
        """Initialize a run-scoped provider factory."""
        self.config = config
        self._sandbox_provider: SandboxProvider | None = None
        self._sandbox_factory = SandboxProviderFactory(
            SandboxProviderConfig(
                providers=config.sandbox_providers,
                object_stores=config.object_stores,
                default_provider=config.default_sandbox_provider,
            )
        )
        self._gateway_sessions: GatewaySessionProvider | None = None

    def create_gateway_sessions(self) -> GatewaySessionProvider:
        """Create or return the run-scoped gateway session provider."""
        if self._gateway_sessions is not None:
            return self._gateway_sessions
        providers: dict[str, GatewaySessionProvider] = {}
        sandbox_provider = self.create_sandbox_provider(
            [config.sandbox_provider.name for config in self.config.kgb_session_configs.values()]
        )
        for agent_name, config in self.config.kgb_session_configs.items():
            providers[agent_name] = KGBGatewaySessionProvider(
                config,
                sandbox_provider=sandbox_provider,
            )
        if not providers:
            self._gateway_sessions = NoopGatewaySessionProvider()
        else:
            self._gateway_sessions = GatewaySessionProviderRouter(
                providers,
                agent_spec_names=self.config.agent_spec_names,
            )
        return self._gateway_sessions

    def close(self) -> None:
        """Close the cached sandbox provider, when present."""
        provider = self._sandbox_provider
        self._sandbox_provider = None
        if provider is not None:
            close = getattr(provider, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> RuntimeProviderFactory:
        """Return this factory as a context manager."""
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        """Close run-scoped providers when leaving the context."""
        self.close()

    def create_sandbox_provider(self, provider_names: Iterable[str] = ()) -> SandboxProvider:
        """Create or return the run-scoped sandbox provider router."""
        if self._sandbox_provider is None:
            self._sandbox_provider = self._sandbox_factory.create(provider_names)
        return self._sandbox_provider

    def create_executor(
        self,
        *,
        output_root: str | Path,
        bus: EventBus,
    ) -> AsyncTaskExecutor:
        """Create an attempt executor from run-scoped services.

        Args:
            output_root: Durable run output root.
            bus: Event bus shared by the run.

        Returns:
            An executor whose attempt decisions come only from requests.

        """
        output_path = Path(output_root)
        provider_names: list[str] = []
        provider_names.extend(config.sandbox_provider.name for config in self.config.kgb_session_configs.values())
        sandbox_provider = self.create_sandbox_provider(provider_names)
        return AsyncTaskExecutor(
            sandbox_provider=sandbox_provider,
            bus=bus,
            output_publisher=LocalOutputPublisher(output_path),
            run_store=LocalRunStore(output_path),
            cleanup_backlog=CleanupBacklog(output_path),
            gateway_sessions=self.create_gateway_sessions(),
        )


def sandbox_provider_names_from_plan(plan: RunPlan) -> list[str]:
    """Return every sandbox provider a run plan can address."""
    names: list[str] = []
    for agent in plan.agents.values():
        names.extend([agent.sandbox.provider, *(sidecar.provider for sidecar in agent.sidecars)])
        raw_verify = agent.sandbox.verify or {}
        raw_verifier = raw_verify.get("sandbox")
        if isinstance(raw_verifier, dict):
            names.append(str(raw_verifier.get("provider") or agent.sandbox.provider))
    return _stable_non_empty(names)


def sandbox_launch_specs_from_plan(plan: RunPlan) -> list[SandboxLaunchSpec]:
    """Compile unique sandbox launch specs used by preflight checks.

    Args:
        plan: Resolved run plan.

    Returns:
        Unique main, sidecar, and verifier launches in stable task order.

    """
    launches: list[SandboxLaunchSpec] = []
    seen: set[str] = set()
    for item in sandbox_launches_by_agent_from_plan(plan):
        key = repr(item.launch.to_dict())
        if key in seen:
            continue
        seen.add(key)
        launches.append(item.launch)
    return launches


def sandbox_launches_by_agent_from_plan(plan: RunPlan) -> list[AgentSandboxLaunch]:
    """Compile unique sandbox launches for each Agent in stable task order."""

    launches: list[AgentSandboxLaunch] = []
    seen: set[tuple[str, str]] = set()
    for task in plan.tasks:
        main = compile_main_sandbox_plan(plan, task)
        candidates = [main.launch]
        candidates.extend(sidecar.launch for sidecar in compile_sidecar_plans(plan, task))
        evaluation = compile_evaluation_plan(plan, task, main)
        if evaluation.sandbox is not None:
            candidates.append(evaluation.sandbox.launch)
        for launch in candidates:
            if launch is None:
                continue
            key = (task.agent_name, repr(launch.to_dict()))
            if key in seen:
                continue
            seen.add(key)
            launches.append(
                AgentSandboxLaunch(
                    agent_name=task.agent_name,
                    agent_spec_name=plan.agent(task.agent_name).agent_spec.name,
                    launch=launch,
                )
            )
    return launches


def _stable_non_empty(values: Iterable[str]) -> list[str]:
    """Return non-empty strings in first-seen order."""
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _sandbox_provider_from_spec(
    spec: SandboxProviderSpec,
    *,
    object_stores: dict[str, ObjectStoreSpec],
) -> SandboxProvider:
    if spec.driver != "akernel":
        raise ValueError(f"unsupported sandbox provider driver: {spec.driver}")
    return AkernelSandboxProvider(spec=spec, object_stores=object_stores)
