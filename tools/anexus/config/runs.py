"""Run-plan resolution from AgentSpec configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from anexus.config.agents import (
    AgentSpec,
    SandboxSpec,
    SidecarSpec,
)
from anexus.config.catalog import (
    AgentRuntimeSpec,
    Catalog,
    ClusterSpec,
    GatewaySpec,
    ObjectStoreSpec,
    SandboxProviderSpec,
)
from anexus.config.loader import ConfigLoadError, load_task_payloads
from anexus.config.uri import resolve_uri
from anexus.outputs.objects import ObjectBinding
from anexus.sandbox.kubeconfig import load_kubeconfig_environment, require_kubeconfig_path
from anexus.tasks import ArtifactSpec, AttemptPolicy, FixturePlan, FixtureSpec, TaskSpec


@dataclass(frozen=True)
class RunOverrides:
    """Explicit run-level overrides, usually from CLI or API input."""

    task_ids: set[str] | None = None
    image: str | None = None
    cpu_request: int | None = None
    cpu_limit: int | None = None
    memory_request_mb: int | None = None
    memory_limit_mb: int | None = None
    task_timeout_seconds: float | None = None
    task_retry_min_wait_seconds: float | None = None
    task_retry_max_wait_seconds: float | None = None
    task_retry_wait_multiplier: float | None = None
    model_id: str | None = None
    judge_model_id: str | None = None
    output_prefix: str | None = None
    repeats: int = 1
    n_concurrent: int = 10
    n_attempts: int = 1
    agent_concurrency_limits: dict[str, int] = field(default_factory=dict)
    agent_concurrency_groups: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate run-level scheduling and retry overrides."""
        if self.repeats < 1:
            raise ValueError("RunOverrides.repeats must be >= 1")
        if self.n_concurrent < 1:
            raise ValueError("RunOverrides.n_concurrent must be >= 1")
        if self.n_attempts < 1:
            raise ValueError("RunOverrides.n_attempts must be >= 1")
        _validate_retry_waits(
            min_wait_seconds=self.task_retry_min_wait_seconds,
            max_wait_seconds=self.task_retry_max_wait_seconds,
            wait_multiplier=self.task_retry_wait_multiplier,
        )
        _validate_agent_concurrency_limits(
            self.agent_concurrency_limits,
            n_concurrent=self.n_concurrent,
        )
        _validate_agent_concurrency_groups(self.agent_concurrency_groups)


@dataclass(frozen=True)
class ResolvedAgentPlan:
    """Catalog decisions used by every task assigned to one Agent."""

    agent_name: str
    agent_spec: AgentSpec
    sandbox: SandboxSpec
    sidecars: list[SidecarSpec]
    model_id: str
    judge_model_id: str
    output_prefix: str
    agent_runtime: AgentRuntimeSpec
    gateway: GatewaySpec
    agent_models: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunPlan:
    """Fully resolved multi-Agent run inputs before resources are created."""

    run_id: str
    tasks: list[TaskSpec]
    agents: dict[str, ResolvedAgentPlan]
    cluster: ClusterSpec
    sandbox_providers: dict[str, SandboxProviderSpec]
    object_stores: dict[str, ObjectStoreSpec]
    n_concurrent: int
    n_attempts: int
    agent_concurrency_limits: dict[str, int]
    agent_concurrency_groups: dict[str, str]
    loaded_task_source: str
    catalog_uri: str = ""
    catalog_version: str = ""

    @property
    def agent_names(self) -> tuple[str, ...]:
        """Return Agent names in stable catalog-resolution order."""
        return tuple(self.agents)

    def agent(self, agent_name: str) -> ResolvedAgentPlan:
        """Return the resolved plan selected by a task row."""
        try:
            return self.agents[agent_name]
        except KeyError as exc:
            known = ", ".join(self.agents) or "<none>"
            raise ValueError(f"task references unresolved agent {agent_name!r}; known agents: {known}") from exc


def resolve_run_plan(
    *,
    catalog: Catalog,
    run_id: str,
    task_payloads: list[dict[str, Any]] | None = None,
    task_source_uri: str | None = None,
    catalog_uri: str = "",
    overrides: RunOverrides | None = None,
    source_uri: str = "memory://run-plan",
    version: int = 1,
) -> RunPlan:
    """Resolve a task source into one run containing any number of Agents."""
    overrides = overrides or RunOverrides()
    effective_source_uri = resolve_uri(catalog_uri, task_source_uri) if task_source_uri else ""
    payloads = task_payloads
    if payloads is None:
        if not effective_source_uri:
            raise ConfigLoadError(
                source_uri,
                "a local task JSONL path is required",
            )
        payloads = _load_task_payloads_from_uri(
            effective_source_uri,
            task_ids=overrides.task_ids,
        )
        source_uri = effective_source_uri
    elif overrides.task_ids:
        payloads = [payload for payload in payloads if str(payload.get("task_id")) in overrides.task_ids]
        if not payloads:
            requested = ", ".join(sorted(overrides.task_ids))
            raise ConfigLoadError(source_uri, f"no matching tasks found for: {requested}")
    if not payloads:
        raise ConfigLoadError(source_uri, "run plan did not contain any tasks")

    agent_names = _agent_names_from_payloads(payloads, source_uri=source_uri)
    agents: dict[str, ResolvedAgentPlan] = {}
    sandbox_providers: dict[str, SandboxProviderSpec] = {}
    for agent_name in agent_names:
        agent_spec = _require_agent(catalog, agent_name)
        runtime = catalog.resources.agent_runtimes[agent_spec.runtime]
        gateway = catalog.resources.gateways[agent_spec.gateway]
        agent_spec = replace(agent_spec, env={**runtime.environment, **agent_spec.env})
        sandbox = _resolve_sandbox(agent_spec, overrides, catalog_uri=catalog_uri)
        agent_models = dict(agent_spec.agent_models)
        model_id = overrides.model_id or agent_models.get("main") or agent_spec.model_id
        agent_models["main"] = model_id
        agents[agent_name] = ResolvedAgentPlan(
            agent_name=agent_name,
            agent_spec=agent_spec,
            sandbox=sandbox,
            sidecars=_resolve_sidecars(agent_spec.sidecars, catalog_uri=catalog_uri),
            model_id=model_id,
            judge_model_id=overrides.judge_model_id or agent_spec.judge_model_id,
            agent_models=agent_models,
            output_prefix=overrides.output_prefix or agent_spec.output_prefix,
            agent_runtime=runtime,
            gateway=gateway,
        )
        sandbox_providers.update(_resolved_sandbox_providers(catalog, agent_spec, gateway))

    tasks = _build_task_specs_for_agents(
        run_id=run_id,
        source_uri=source_uri,
        payloads=payloads,
        agents=agents,
        repeats=overrides.repeats,
        overrides=overrides,
        version=version,
        catalog_version=catalog.version,
        cluster=catalog.cluster,
        catalog_uri=catalog_uri,
    )
    object_stores: dict[str, ObjectStoreSpec] = {}
    for agent_name, agent in agents.items():
        object_stores.update(
            _resolved_object_stores(
                catalog,
                agent.agent_spec,
                agent.gateway,
                [task for task in tasks if task.agent_name == agent_name],
            )
        )
    return RunPlan(
        run_id=run_id,
        tasks=tasks,
        agents=agents,
        cluster=catalog.cluster,
        sandbox_providers=sandbox_providers,
        object_stores=object_stores,
        n_concurrent=overrides.n_concurrent,
        n_attempts=overrides.n_attempts,
        agent_concurrency_limits=_resolve_all_agent_concurrency_limits(
            agents=agents,
            overrides=overrides,
        ),
        agent_concurrency_groups=dict(overrides.agent_concurrency_groups),
        loaded_task_source=effective_source_uri,
        catalog_uri=catalog_uri,
        catalog_version=catalog.version,
    )


def task_from_run_plan(
    plan: RunPlan,
    payload: dict[str, Any],
    *,
    source_uri: str,
    labels: dict[str, str] | None = None,
    version: int = 1,
) -> TaskSpec:
    """Build one task using the catalog decisions in a resolved run plan.

    This is the task-loading boundary used by dynamic RL and swarm inputs. It
    preserves catalog bindings, retry policy, and AgentSpec identity before the
    runtime compiler creates an execution request.

    Args:
        plan: Resolved run plan supplying agent and catalog decisions.
        payload: Dynamic task payload.
        source_uri: Source identifier recorded on the task.
        labels: Additional task labels.
        version: Task schema version.

    Returns:
        A task with payload fixtures and catalog bindings merged once.

    """
    agent_name = str(payload.get("agent_name", "")).strip()
    agent = plan.agent(agent_name)
    matching_tasks = [task for task in plan.tasks if task.agent_name == agent_name]
    if matching_tasks:
        attempt_policy = replace(matching_tasks[0].attempt_policy, repeat_id=None)
        task_labels = dict(matching_tasks[0].labels)
    else:
        attempt_policy = AttemptPolicy(timeout_seconds=agent.agent_spec.task_timeout_seconds)
        task_labels = {"agent_spec": agent.agent_spec.name}
    task_labels.update(labels or {})
    task = TaskSpec.from_payload(
        run_id=plan.run_id,
        agent_name=agent_name,
        payload=payload,
        source_uri=source_uri,
        version=version,
        labels=task_labels,
        attempt_policy=attempt_policy,
    )
    fixture_plan = _fixture_plan_from_bindings(
        agent.sandbox.bindings,
        catalog_uri=plan.catalog_uri,
    )
    binding_artifacts = _artifact_specs_from_bindings(
        agent.sandbox.bindings,
        catalog_uri=plan.catalog_uri,
    )
    if not fixture_plan.fixtures and not binding_artifacts:
        return task
    return replace(
        task,
        fixtures=FixturePlan([*task.fixtures.fixtures, *fixture_plan.fixtures]),
        artifacts=[*task.artifacts, *binding_artifacts],
    )


def _agent_names_from_payloads(
    payloads: list[dict[str, Any]],
    *,
    source_uri: str,
) -> list[str]:
    names: list[str] = []
    for index, payload in enumerate(payloads, start=1):
        agent_name = str(payload.get("agent_name", "")).strip()
        if not agent_name:
            raise ConfigLoadError(
                source_uri,
                "task payload requires 'agent_name'",
                index,
            )
        if agent_name not in names:
            names.append(agent_name)
    return names


def _resolve_all_agent_concurrency_limits(
    *,
    agents: dict[str, ResolvedAgentPlan],
    overrides: RunOverrides,
) -> dict[str, int]:
    limits = dict(overrides.agent_concurrency_limits)
    for agent_name, agent in agents.items():
        pool_name = overrides.agent_concurrency_groups.get(agent_name, agent_name)
        configured = agent.agent_spec.n_concurrent
        if configured is not None and pool_name not in limits:
            limits[pool_name] = configured
        elif configured is not None and limits.get(pool_name) != configured:
            raise ValueError(f"agents sharing concurrency pool {pool_name!r} declare different limits")
        if agent_name in overrides.agent_concurrency_groups and pool_name not in limits:
            raise ValueError(
                f"agent concurrency group {pool_name!r} for {agent_name!r} requires an "
                "agent concurrency limit from the run override or AgentSpec"
            )
    _validate_agent_concurrency_limits(limits, n_concurrent=overrides.n_concurrent)
    return limits


def _validate_agent_concurrency_limits(
    limits: dict[str, int],
    *,
    n_concurrent: int,
) -> None:
    for agent_name, limit in limits.items():
        if not agent_name:
            raise ValueError("agent concurrency limit name is required")
        if limit < 1:
            raise ValueError(f"agent concurrency limit for {agent_name!r} must be >= 1")
        if limit > n_concurrent:
            raise ValueError(f"agent concurrency limit for {agent_name!r} must not exceed n_concurrent")


def _validate_agent_concurrency_groups(groups: dict[str, str]) -> None:
    for agent_name, group_name in groups.items():
        if not agent_name:
            raise ValueError("agent concurrency group agent name is required")
        if not group_name:
            raise ValueError(f"agent concurrency group for {agent_name!r} is required")


def _validate_retry_waits(
    *,
    min_wait_seconds: float | None,
    max_wait_seconds: float | None,
    wait_multiplier: float | None,
) -> None:
    if min_wait_seconds is not None and min_wait_seconds < 0:
        raise ValueError("task_retry_min_wait_seconds must be >= 0")
    if max_wait_seconds is not None and max_wait_seconds < 0:
        raise ValueError("task_retry_max_wait_seconds must be >= 0")
    if wait_multiplier is not None and wait_multiplier < 1:
        raise ValueError("task_retry_wait_multiplier must be >= 1")


def _override_float(value: float | None, fallback: float) -> float:
    return fallback if value is None else value


def _require_agent(catalog: Catalog, agent_name: str) -> AgentSpec:
    agent_spec = catalog.get(agent_name)
    if agent_spec is None:
        known = ", ".join(catalog.names()) or "<none>"
        raise KeyError(f"unknown agent {agent_name!r}; known agents: {known}")
    if agent_spec.sandbox is None:
        raise ValueError(f"agent {agent_name!r} does not define a sandbox")
    return agent_spec


def _resolved_sandbox_providers(
    catalog: Catalog,
    agent_spec: AgentSpec,
    gateway: GatewaySpec,
) -> dict[str, SandboxProviderSpec]:
    assert agent_spec.sandbox is not None
    names = {
        agent_spec.sandbox.provider,
        gateway.sandbox_provider,
        *(sidecar.provider for sidecar in agent_spec.sidecars),
    }
    raw_verify = agent_spec.sandbox.verify or {}
    raw_verifier = raw_verify.get("sandbox")
    if isinstance(raw_verifier, dict):
        names.add(str(raw_verifier.get("provider") or agent_spec.sandbox.provider))
    providers = {name: catalog.resources.sandbox_providers[name] for name in names}
    control_planes = {provider.control_plane for provider in providers.values()}
    if len(control_planes) > 1:
        raise ValueError("one run cannot use AKernel providers with different control planes")
    return providers


def _resolved_object_stores(
    catalog: Catalog,
    agent_spec: AgentSpec,
    gateway: GatewaySpec,
    tasks: list[TaskSpec],
) -> dict[str, ObjectStoreSpec]:
    names = {gateway.trace.object_store}
    if agent_spec.sandbox is not None:
        for binding in agent_spec.sandbox.bindings:
            parts = urlsplit(binding.object_uri)
            if parts.scheme != "oss":
                continue
            matches = [name for name, store in catalog.resources.object_stores.items() if store.bucket == parts.netloc]
            if len(matches) != 1:
                raise ValueError(f"binding OSS bucket {parts.netloc!r} must match exactly one object store")
            names.add(matches[0])
    task_uris = [
        *(fixture.source_uri for task in tasks for fixture in task.fixtures.fixtures),
        *(artifact.uri for task in tasks for artifact in task.artifacts),
    ]
    for uri in task_uris:
        parts = urlsplit(uri)
        if parts.scheme != "oss":
            continue
        matches = [name for name, store in catalog.resources.object_stores.items() if store.bucket == parts.netloc]
        if len(matches) != 1:
            raise ValueError(f"task OSS bucket {parts.netloc!r} must match exactly one object store")
        names.add(matches[0])
    return {name: catalog.resources.object_stores[name] for name in names}


def _resolve_sandbox(
    agent_spec: AgentSpec,
    overrides: RunOverrides,
    *,
    catalog_uri: str,
) -> SandboxSpec:
    assert agent_spec.sandbox is not None
    sandbox = agent_spec.sandbox
    return replace(
        sandbox,
        image=overrides.image or sandbox.image,
        cpu_request=overrides.cpu_request or sandbox.cpu_request,
        cpu_limit=overrides.cpu_limit or overrides.cpu_request or sandbox.cpu_limit,
        memory_request_mb=overrides.memory_request_mb or sandbox.memory_request_mb,
        memory_limit_mb=overrides.memory_limit_mb or overrides.memory_request_mb or sandbox.memory_limit_mb,
        file_overrides=_resolve_file_overrides(sandbox.file_overrides, catalog_uri=catalog_uri),
    )


def _resolve_file_overrides(
    file_overrides: list[dict[str, str]],
    *,
    catalog_uri: str,
) -> list[dict[str, str]]:
    return [
        {
            **override,
            "src": _resolve_local_config_path(catalog_uri, override["src"]),
        }
        for override in file_overrides
        if "src" in override
    ]


def _resolve_sidecars(sidecars: list[SidecarSpec], *, catalog_uri: str) -> list[SidecarSpec]:
    resolved: list[SidecarSpec] = []
    for sidecar in sidecars:
        kubeconfig = _resolve_local_config_path(catalog_uri, sidecar.kubeconfig) if sidecar.kubeconfig else ""
        env = dict(sidecar.env)
        if sidecar.inject_kubeconfig:
            env.update(load_kubeconfig_environment(require_kubeconfig_path(kubeconfig)))
        resolved.append(replace(sidecar, kubeconfig=kubeconfig, env=env))
    return resolved


def _resolve_local_config_path(catalog_uri: str, value: str) -> str:
    resolved = resolve_uri(catalog_uri, value)
    parts = urlsplit(resolved)
    if parts.scheme == "file":
        return str(Path(unquote(parts.path)))
    return resolved


def _build_task_specs_for_agents(
    *,
    run_id: str,
    source_uri: str,
    payloads: list[dict[str, Any]],
    agents: dict[str, ResolvedAgentPlan],
    repeats: int,
    overrides: RunOverrides,
    version: int,
    catalog_version: str,
    cluster: ClusterSpec,
    catalog_uri: str,
) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for repeat_index in range(repeats):
        repeat_id = str(repeat_index) if repeats > 1 else None
        for payload in payloads:
            payload_copy = dict(payload)
            agent_name = str(payload_copy.get("agent_name", "")).strip()
            agent = agents[agent_name]
            fixture_plan = _fixture_plan_from_bindings(
                agent.sandbox.bindings,
                catalog_uri=catalog_uri,
            )
            binding_artifacts = _artifact_specs_from_bindings(
                agent.sandbox.bindings,
                catalog_uri=catalog_uri,
            )
            if repeat_id is not None:
                payload_copy["_repeat_id"] = repeat_id
            task = TaskSpec.from_payload(
                run_id=run_id,
                agent_name=agent_name,
                payload=payload_copy,
                source_uri=source_uri,
                version=version,
                labels={
                    "agent_spec": agent.agent_spec.name,
                    "catalog_version": catalog_version,
                    "cluster": cluster.name,
                    "region": cluster.region,
                },
                attempt_policy=AttemptPolicy(
                    timeout_seconds=(overrides.task_timeout_seconds or agent.agent_spec.task_timeout_seconds),
                    repeat_id=repeat_id,
                    min_retry_wait_seconds=_override_float(
                        overrides.task_retry_min_wait_seconds,
                        agent.agent_spec.min_retry_wait_seconds,
                    ),
                    max_retry_wait_seconds=_override_float(
                        overrides.task_retry_max_wait_seconds,
                        agent.agent_spec.max_retry_wait_seconds,
                    ),
                    retry_wait_multiplier=_override_float(
                        overrides.task_retry_wait_multiplier,
                        agent.agent_spec.retry_wait_multiplier,
                    ),
                ),
            )
            if fixture_plan.fixtures or binding_artifacts:
                task = replace(
                    task,
                    fixtures=FixturePlan([*task.fixtures.fixtures, *fixture_plan.fixtures]),
                    artifacts=[*task.artifacts, *binding_artifacts],
                )
            tasks.append(task)
    return tasks


def _fixture_plan_from_bindings(
    bindings: list[ObjectBinding],
    *,
    catalog_uri: str,
) -> FixturePlan:
    fixtures: list[FixtureSpec] = []
    for binding in bindings:
        if not binding.reads:
            continue
        uri = resolve_uri(catalog_uri, binding.object_uri)
        fixtures.append(
            FixtureSpec(
                source_path="",
                source_uri=uri,
                target_path=_binding_local_path(binding.target_path),
                kind=binding.kind,
                required=not binding.optional,
                sandbox_path=binding.target_path,
            )
        )
    return FixturePlan(fixtures)


def _artifact_specs_from_bindings(
    bindings: list[ObjectBinding],
    *,
    catalog_uri: str,
) -> list[ArtifactSpec]:
    artifacts: list[ArtifactSpec] = []
    for index, binding in enumerate(bindings):
        if not binding.writes:
            continue
        local_path = f"fixtures/{_binding_local_path(binding.target_path)}"
        artifacts.append(
            ArtifactSpec(
                object_id=f"binding:{index}:{_binding_object_part(binding.target_path)}",
                kind=f"binding.{binding.kind}",
                local_path=local_path,
                uri=resolve_uri(catalog_uri, binding.object_uri),
                required=not binding.optional,
                source_path=binding.target_path,
                storage_kind=binding.kind,
            )
        )
    return artifacts


def _binding_local_path(target_path: str) -> str:
    parts = _binding_target_parts(target_path)
    return "bindings/" + "/".join(parts)


def _binding_object_part(target_path: str) -> str:
    return "-".join(_binding_target_parts(target_path))


def _binding_target_parts(target_path: str) -> list[str]:
    parts = [part for part in target_path.replace("\\", "/").strip().split("/") if part and part != "."]
    if not parts:
        return ["root"]
    if any(part == ".." for part in parts):
        raise ValueError(f"binding target escapes workspace: {target_path}")
    return parts


def _local_path_from_uri(uri: str) -> Path:
    parts = urlsplit(uri)
    if parts.scheme in {"", "file"}:
        return Path(unquote(parts.path))
    raise ConfigLoadError(uri, "only local task source URIs can be loaded directly")


def _load_task_payloads_from_uri(
    uri: str,
    *,
    task_ids: set[str] | None,
) -> list[dict[str, Any]]:
    parts = urlsplit(uri)
    if parts.scheme in {"", "file"}:
        return load_task_payloads(_local_path_from_uri(uri), task_ids=task_ids)
    raise ConfigLoadError(uri, "task JSONL must be a local file")
