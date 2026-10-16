"""Run-plan preflight checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

from anexus.config import RunPlan, SandboxSpec, SidecarSpec
from anexus.core.context import TaskContext
from anexus.core.errors import AnexusError
from anexus.gateway import ModelRouter, alias_target_model
from anexus.outputs import ObjectStore
from anexus.runtime.capacity import estimate_run_capacity
from anexus.sandbox import (
    SandboxProvider,
    check_sandbox_provider,
    read_sandbox_provider_resource_snapshot,
)
from anexus.sandbox.kubeconfig import load_kubeconfig_environment, require_kubeconfig_path


@dataclass(frozen=True)
class PreflightDiagnostic:
    """Describe one preflight finding."""

    level: Literal["ok", "warn", "error"]
    category: str
    message: str
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable diagnostic."""
        return asdict(self)


@dataclass(frozen=True)
class PreflightReport:
    """Summarize preflight findings for one run plan."""

    run_id: str
    agent_names: list[str]
    task_count: int
    diagnostics: list[PreflightDiagnostic] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return whether any diagnostic is an error."""
        return any(diagnostic.level == "error" for diagnostic in self.diagnostics)

    @property
    def error_count(self) -> int:
        """Return the number of error diagnostics."""
        return sum(1 for diagnostic in self.diagnostics if diagnostic.level == "error")

    @property
    def warning_count(self) -> int:
        """Return the number of warning diagnostics."""
        return sum(1 for diagnostic in self.diagnostics if diagnostic.level == "warn")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report."""
        return {
            "run_id": self.run_id,
            "agent_names": list(self.agent_names),
            "task_count": self.task_count,
            "status": "failed" if self.has_errors else "ok",
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


async def preflight_run_plan(
    plan: RunPlan,
    *,
    object_store: ObjectStore | None = None,
    sandbox_provider: SandboxProvider | None = None,
    check_object_fixtures: bool = True,
) -> PreflightReport:
    """Validate a resolved run plan before allocating runtime resources."""
    diagnostics: list[PreflightDiagnostic] = []
    diagnostics.append(
        PreflightDiagnostic(
            "ok",
            "plan",
            f"resolved {len(plan.tasks)} task(s) for {len(plan.agents)} agent(s)",
            detail=f"source={plan.loaded_task_source}",
        )
    )
    if not plan.tasks:
        diagnostics.append(PreflightDiagnostic("error", "tasks", "run plan has no tasks"))
    for agent_name, agent in plan.agents.items():
        diagnostics.extend(_sandbox_diagnostics(agent.sandbox))
        diagnostics.extend(_sidecar_diagnostics(agent.sidecars))
        diagnostics.extend(_gateway_diagnostics(plan, agent_name))
    diagnostics.extend(_capacity_diagnostics(plan))

    diagnostics.extend(
        _fixture_diagnostics(
            plan,
            object_store=object_store,
            check_object_fixtures=check_object_fixtures,
        )
    )
    diagnostics.extend(_artifact_diagnostics(plan))
    if sandbox_provider is not None:
        diagnostics.extend(await _sandbox_resource_diagnostics(plan, sandbox_provider=sandbox_provider))
        diagnostics.extend(await _sandbox_health_diagnostics(plan, sandbox_provider=sandbox_provider))
    return PreflightReport(
        run_id=plan.run_id,
        agent_names=list(plan.agent_names),
        task_count=len(plan.tasks),
        diagnostics=diagnostics,
    )


def _sandbox_diagnostics(sandbox: SandboxSpec) -> list[PreflightDiagnostic]:
    diagnostics: list[PreflightDiagnostic] = []
    if not sandbox.provider:
        diagnostics.append(PreflightDiagnostic("error", "sandbox", "sandbox provider name is required"))
    if not sandbox.image:
        diagnostics.append(PreflightDiagnostic("error", "sandbox", "sandbox image is required"))
    else:
        diagnostics.append(
            PreflightDiagnostic(
                "ok",
                "sandbox",
                f"{sandbox.provider} image={sandbox.image}",
            )
        )
    diagnostics.extend(
        _resource_diagnostics(
            category="sandbox",
            label="main sandbox",
            port=sandbox.port,
            cpu_request=sandbox.cpu_request,
            cpu_limit=sandbox.cpu_limit,
            memory_request_mb=sandbox.memory_request_mb,
            memory_limit_mb=sandbox.memory_limit_mb,
        )
    )
    return diagnostics


def _sidecar_diagnostics(sidecars: list[SidecarSpec]) -> list[PreflightDiagnostic]:
    diagnostics: list[PreflightDiagnostic] = []
    seen_names: set[str] = set()
    for index, sidecar in enumerate(sidecars):
        label = sidecar.name or f"sidecar[{index}]"
        if not sidecar.name:
            diagnostics.append(PreflightDiagnostic("error", "sidecars", f"sidecar name is required: {label}"))
        elif sidecar.name in seen_names:
            diagnostics.append(PreflightDiagnostic("error", "sidecars", f"duplicate sidecar name: {sidecar.name}"))
        seen_names.add(sidecar.name)

        if not sidecar.provider:
            diagnostics.append(PreflightDiagnostic("error", "sidecars", f"sidecar provider is required: {label}"))
        if not sidecar.image:
            diagnostics.append(PreflightDiagnostic("error", "sidecars", f"sidecar image is required: {label}"))
        else:
            diagnostics.append(
                PreflightDiagnostic(
                    "ok",
                    "sidecars",
                    f"{label} provider={sidecar.provider} image={sidecar.image}",
                )
            )
        if sidecar.health_timeout_seconds < 0:
            diagnostics.append(
                PreflightDiagnostic(
                    "error",
                    "sidecars",
                    f"{label} health_timeout_seconds must be >= 0",
                )
            )
        elif sidecar.health_timeout_seconds:
            diagnostics.append(
                PreflightDiagnostic(
                    "ok",
                    "sidecars",
                    f"{label} health check enabled",
                    detail=(
                        f"health_path={sidecar.health_path or '/'} "
                        f"health_timeout_seconds={sidecar.health_timeout_seconds}"
                    ),
                )
            )
        elif sidecar.health_path:
            diagnostics.append(
                PreflightDiagnostic(
                    "warn",
                    "sidecars",
                    f"{label} health_path is ignored without health_timeout_seconds",
                    detail=f"health_path={sidecar.health_path}",
                )
            )
        diagnostics.extend(_sidecar_kubeconfig_diagnostics(sidecar, label=label))
        if sidecar.ports:
            invalid_ports = [port for port in sidecar.ports if port < 1 or port > 65535]
            for port in invalid_ports:
                diagnostics.append(
                    PreflightDiagnostic(
                        "error",
                        "sidecars",
                        f"invalid forwarded port for {label}: {port}",
                    )
                )
            if sidecar.port not in sidecar.ports:
                diagnostics.append(
                    PreflightDiagnostic(
                        "error",
                        "sidecars",
                        f"{label} primary port must be included in ports",
                        detail=f"port={sidecar.port} ports={sidecar.ports}",
                    )
                )
            elif not invalid_ports:
                diagnostics.append(
                    PreflightDiagnostic(
                        "ok",
                        "sidecars",
                        f"{label} forwards {len(sidecar.ports)} port(s)",
                        detail=f"ports={sidecar.ports}",
                    )
                )
        diagnostics.extend(
            _resource_diagnostics(
                category="sidecars",
                label=label,
                port=sidecar.port,
                cpu_request=sidecar.cpu_request,
                cpu_limit=sidecar.cpu_limit,
                memory_request_mb=sidecar.memory_request_mb,
                memory_limit_mb=sidecar.memory_limit_mb,
            )
        )
    return diagnostics


def _sidecar_kubeconfig_diagnostics(
    sidecar: SidecarSpec,
    *,
    label: str,
) -> list[PreflightDiagnostic]:
    if not sidecar.inject_kubeconfig:
        if sidecar.kubeconfig:
            return [
                PreflightDiagnostic(
                    "warn",
                    "sidecars",
                    f"{label} kubeconfig is ignored without inject_kubeconfig",
                    detail=f"kubeconfig={sidecar.kubeconfig}",
                )
            ]
        return []
    try:
        path = require_kubeconfig_path(sidecar.kubeconfig)
        env = load_kubeconfig_environment(path)
    except (OSError, ValueError) as exc:
        return [
            PreflightDiagnostic(
                "error",
                "sidecars",
                f"{label} kubeconfig injection is invalid",
                detail=str(exc),
            )
        ]
    return [
        PreflightDiagnostic(
            "ok",
            "sidecars",
            f"{label} kubeconfig injection enabled",
            detail=f"kubeconfig={sidecar.kubeconfig} env_keys={sorted(env)}",
        )
    ]


def _resource_diagnostics(
    *,
    category: str,
    label: str,
    port: int,
    cpu_request: int,
    cpu_limit: int,
    memory_request_mb: int,
    memory_limit_mb: int,
) -> list[PreflightDiagnostic]:
    diagnostics: list[PreflightDiagnostic] = []
    if port < 1 or port > 65535:
        diagnostics.append(PreflightDiagnostic("error", category, f"invalid port for {label}: {port}"))
    if cpu_request <= 0:
        diagnostics.append(
            PreflightDiagnostic(
                "error",
                category,
                f"cpu_request must be positive for {label}: {cpu_request}",
            )
        )
    if cpu_limit <= 0:
        diagnostics.append(PreflightDiagnostic("error", category, f"cpu_limit must be positive for {label}"))
    if cpu_request > 0 and cpu_limit > 0 and cpu_request > cpu_limit:
        diagnostics.append(
            PreflightDiagnostic(
                "error",
                category,
                f"cpu_request exceeds cpu_limit for {label}",
                detail=f"{cpu_request}>{cpu_limit}",
            )
        )
    if memory_request_mb <= 0:
        diagnostics.append(
            PreflightDiagnostic(
                "error",
                category,
                f"memory_request_mb must be positive for {label}: {memory_request_mb}",
            )
        )
    if memory_limit_mb <= 0:
        diagnostics.append(
            PreflightDiagnostic(
                "error",
                category,
                f"memory_limit_mb must be positive for {label}: {memory_limit_mb}",
            )
        )
    if memory_request_mb > 0 and memory_limit_mb > 0 and memory_request_mb > memory_limit_mb:
        diagnostics.append(
            PreflightDiagnostic(
                "error",
                category,
                f"memory_request_mb exceeds memory_limit_mb for {label}",
                detail=f"{memory_request_mb}>{memory_limit_mb}",
            )
        )
    return diagnostics


def _fixture_diagnostics(
    plan: RunPlan,
    *,
    object_store: ObjectStore | None,
    check_object_fixtures: bool,
) -> list[PreflightDiagnostic]:
    diagnostics: list[PreflightDiagnostic] = []
    seen: set[tuple[str, str, str, bool]] = set()
    for task in plan.tasks:
        for fixture in task.fixtures.fixtures:
            key = (
                task.agent_name,
                fixture.source_path or fixture.source_uri,
                fixture.target_path,
                fixture.required,
            )
            if key in seen:
                continue
            seen.add(key)
            if fixture.source_uri:
                if check_object_fixtures:
                    diagnostics.append(
                        _object_fixture_diagnostic(
                            fixture.source_uri,
                            required=fixture.required,
                            object_store=object_store,
                        )
                    )
                else:
                    diagnostics.append(
                        PreflightDiagnostic(
                            "warn",
                            "fixtures",
                            f"object fixture existence was not checked: {fixture.source_uri}",
                            detail="runtime data transfer will validate the object",
                        )
                    )
                continue
            source = Path(fixture.source_path)
            if source.exists():
                diagnostics.append(
                    PreflightDiagnostic(
                        "ok",
                        "fixtures",
                        f"fixture source exists: {source}",
                    )
                )
            elif fixture.required:
                diagnostics.append(
                    PreflightDiagnostic(
                        "error",
                        "fixtures",
                        f"required fixture source is missing: {source}",
                    )
                )
            else:
                diagnostics.append(
                    PreflightDiagnostic(
                        "warn",
                        "fixtures",
                        f"optional fixture source is missing: {source}",
                    )
                )
    return diagnostics


def _capacity_diagnostics(plan: RunPlan) -> list[PreflightDiagnostic]:
    capacity = estimate_run_capacity(plan)
    attempt_detail = f"n_attempts={capacity.attempts_per_task} max_active_attempts={capacity.max_active_attempts}"
    if capacity.agent_concurrency_limit:
        attempt_detail += (
            f" agent_concurrency_limit={capacity.agent_concurrency_limit}"
            f" max_active_agent_phases={capacity.max_active_agent_phases}"
        )
    diagnostics = [
        PreflightDiagnostic(
            "ok",
            "runtime",
            f"scheduled {capacity.total_attempts} attempt(s) for {capacity.task_count} task(s)",
            detail=attempt_detail,
        ),
        PreflightDiagnostic(
            "ok",
            "runtime",
            f"peak sandbox capacity is {capacity.max_active_sandboxes} sandbox(es)",
            detail=(
                f"sandboxes_per_attempt={capacity.sandboxes_per_attempt} "
                f"total_sandbox_creates={capacity.total_sandbox_creates} "
                f"cpu_request={capacity.max_active_cpu_request} "
                f"memory_request_mb={capacity.max_active_memory_request_mb}"
            ),
        ),
    ]
    if capacity.total_attempts and plan.n_concurrent > capacity.total_attempts:
        diagnostics.append(
            PreflightDiagnostic(
                "warn",
                "runtime",
                "n_concurrent exceeds scheduled attempts",
                detail=f"n_concurrent={plan.n_concurrent} total_attempts={capacity.total_attempts}",
            )
        )
    return diagnostics


def _gateway_diagnostics(plan: RunPlan, agent_name: str) -> list[PreflightDiagnostic]:
    agent = plan.agent(agent_name)
    diagnostics: list[PreflightDiagnostic] = []
    if not agent.model_id:
        diagnostics.append(PreflightDiagnostic("warn", "gateway", f"{agent_name}: model id is empty"))

    diagnostics.extend(_mirror_table_diagnostics(plan, agent_name))
    router = ModelRouter(
        mirrors=list(agent.gateway.mirrors),
        strict=True,
    )
    role_models = {
        f"{agent_name}.worker.main": agent.model_id,
        **{f"{agent_name}.worker.{role}": model for role, model in agent.agent_models.items() if role != "main"},
        f"{agent_name}.judge": agent.judge_model_id,
    }
    for role, model in role_models.items():
        if not model:
            continue
        diagnostics.append(_model_route_diagnostic(router, role=role, model=model))
    return diagnostics


def _mirror_table_diagnostics(
    plan: RunPlan,
    agent_name: str,
) -> list[PreflightDiagnostic]:
    agent = plan.agent(agent_name)
    diagnostics: list[PreflightDiagnostic] = []
    seen_models: dict[str, str] = {}
    for mirror in agent.gateway.mirrors:
        if mirror.protocol is None:
            diagnostics.append(
                PreflightDiagnostic(
                    "error",
                    "gateway",
                    f"mirror has no supported upstream: {mirror.name}",
                )
            )
        for model in mirror.models:
            raw_model = alias_target_model(model)
            first_mirror = seen_models.setdefault(raw_model, mirror.name)
            if first_mirror != mirror.name:
                diagnostics.append(
                    PreflightDiagnostic(
                        "error",
                        "gateway",
                        f"model is assigned to multiple mirrors: {raw_model}",
                        detail=f"{first_mirror},{mirror.name}",
                    )
                )
    return diagnostics


def _model_route_diagnostic(
    router: ModelRouter,
    *,
    role: str,
    model: str,
) -> PreflightDiagnostic:
    raw_model = alias_target_model(model)
    try:
        is_engine_model = router.is_engine_model(raw_model)
    except KeyError as exc:
        return PreflightDiagnostic(
            "error",
            "gateway",
            f"{role} model is not registered in engine models or mirrors: {model}",
            detail=str(exc),
        )
    if is_engine_model:
        return PreflightDiagnostic(
            "ok",
            "gateway",
            f"{role} model uses engine gateway: {model}",
            detail=f"target={raw_model}",
        )

    mirror = router.mirror_for_model(raw_model)
    route_key = router.mirror_route_key(raw_model)
    protocol = router.mirror_protocol(raw_model)
    if mirror is None or route_key is None or protocol is None:
        return PreflightDiagnostic(
            "error",
            "gateway",
            f"{role} model mirror route is incomplete: {model}",
            detail=f"target={raw_model}",
        )
    return PreflightDiagnostic(
        "ok",
        "gateway",
        f"{role} model uses mirror {mirror.name}: {model}",
        detail=f"target={raw_model} route={route_key} protocol={protocol}",
    )


def _object_fixture_diagnostic(
    uri: str,
    *,
    required: bool,
    object_store: ObjectStore | None,
) -> PreflightDiagnostic:
    if object_store is None:
        level = "error" if required else "warn"
        return PreflightDiagnostic(level, "fixtures", f"object fixture requires store: {uri}")
    try:
        object_store.stat(uri)
    except AnexusError as exc:
        level = "error" if required else "warn"
        return PreflightDiagnostic(level, "fixtures", str(exc), detail=uri)
    except Exception as exc:
        level = "error" if required else "warn"
        return PreflightDiagnostic(
            level,
            "fixtures",
            str(exc) or exc.__class__.__name__,
            detail=uri,
        )
    return PreflightDiagnostic("ok", "fixtures", f"object fixture exists: {uri}")


def _artifact_diagnostics(plan: RunPlan) -> list[PreflightDiagnostic]:
    diagnostics: list[PreflightDiagnostic] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for task in plan.tasks:
        for artifact in task.artifacts:
            key = (
                task.agent_name,
                artifact.object_id,
                artifact.uri,
                artifact.local_path,
                artifact.source_path,
            )
            if key in seen:
                continue
            seen.add(key)
            if not artifact.uri:
                diagnostics.append(
                    PreflightDiagnostic(
                        "error",
                        "artifacts",
                        f"artifact URI is required: {artifact.object_id}",
                    )
                )
            elif not artifact.local_path:
                diagnostics.append(
                    PreflightDiagnostic(
                        "warn",
                        "artifacts",
                        f"artifact local path is empty: {artifact.object_id}",
                        detail=artifact.uri,
                    )
                )
            else:
                path_diagnostics = _artifact_path_diagnostics(artifact)
                diagnostics.extend(path_diagnostics)
                if path_diagnostics:
                    continue
                diagnostics.append(
                    PreflightDiagnostic(
                        "ok",
                        "artifacts",
                        f"artifact declared: {artifact.object_id}",
                        detail=artifact.uri,
                    )
                )
    return diagnostics


def _artifact_path_diagnostics(artifact) -> list[PreflightDiagnostic]:
    diagnostics: list[PreflightDiagnostic] = []
    local_path = artifact.local_path.strip()
    local_parts = PurePosixPath(local_path).parts
    if "\\" in local_path:
        diagnostics.append(
            _artifact_path_diagnostic(
                artifact,
                field="local_path",
                reason="backslash_path_separator",
            )
        )
    if PurePosixPath(local_path).is_absolute():
        diagnostics.append(
            _artifact_path_diagnostic(
                artifact,
                field="local_path",
                reason="absolute_path_outside_attempt",
            )
        )
    if not local_parts or local_path == ".":
        diagnostics.append(
            _artifact_path_diagnostic(
                artifact,
                field="local_path",
                reason="attempt_root_path",
            )
        )
    if any(part == ".." for part in local_parts):
        diagnostics.append(
            _artifact_path_diagnostic(
                artifact,
                field="local_path",
                reason="path_escapes_attempt",
            )
        )
    if local_parts and local_parts[0] == "manifest.json":
        diagnostics.append(
            _artifact_path_diagnostic(
                artifact,
                field="local_path",
                reason="reserved_artifact_path",
            )
        )
    source_path = artifact.source_path.strip()
    if source_path:
        source_parts = PurePosixPath(source_path).parts
        if "\\" in source_path:
            diagnostics.append(
                _artifact_path_diagnostic(
                    artifact,
                    field="source_path",
                    reason="backslash_path_separator",
                )
            )
        if any(part == ".." for part in source_parts):
            diagnostics.append(
                _artifact_path_diagnostic(
                    artifact,
                    field="source_path",
                    reason="path_contains_parent_reference",
                )
            )
    return diagnostics


def _artifact_path_diagnostic(
    artifact,
    *,
    field: str,
    reason: str,
) -> PreflightDiagnostic:
    return PreflightDiagnostic(
        "error",
        "artifacts",
        f"artifact {field} is unsafe: {artifact.object_id}",
        detail=f"reason={reason} uri={artifact.uri}",
    )


async def _sandbox_health_diagnostics(
    plan: RunPlan,
    *,
    sandbox_provider: SandboxProvider,
) -> list[PreflightDiagnostic]:
    from anexus.api.providers import sandbox_launches_by_agent_from_plan

    diagnostics: list[PreflightDiagnostic] = []
    for index, item in enumerate(sandbox_launches_by_agent_from_plan(plan)):
        spec = item.launch
        role = "main" if spec.name == "main" else spec.name
        context = TaskContext(
            run_id=plan.run_id,
            task_id="preflight",
            version=1,
            attempt_id=f"preflight-{index}-{role}-{spec.name}",
            agent_name=item.agent_name,
            agent_spec_name=item.agent_spec_name,
        ).with_worker_pid()
        report = await check_sandbox_provider(sandbox_provider, context, spec=spec)
        detail = _json_diagnostic_detail(
            {
                "provider": report.provider,
                "agent_name": item.agent_name,
                "role": role,
                "name": spec.name,
                "image": spec.image,
                "sandbox_id": report.sandbox_id,
                "latency_seconds": report.latency_seconds,
                "errors": [error.to_dict() for error in report.errors],
            }
        )
        if report.status == "healthy":
            diagnostics.append(
                PreflightDiagnostic(
                    "ok",
                    "sandbox_health",
                    f"{item.agent_name}: {role} sandbox provider is healthy: {spec.provider}/{spec.name}",
                    detail=detail,
                )
            )
        else:
            diagnostics.append(
                PreflightDiagnostic(
                    "error",
                    "sandbox_health",
                    f"{item.agent_name}: {role} sandbox provider check failed: {spec.provider}/{spec.name}",
                    detail=detail,
                )
            )
    return diagnostics


async def _sandbox_resource_diagnostics(
    plan: RunPlan,
    *,
    sandbox_provider: SandboxProvider,
) -> list[PreflightDiagnostic]:
    try:
        snapshot = await read_sandbox_provider_resource_snapshot(sandbox_provider)
    except (OSError, RuntimeError, TypeError) as exc:
        return [
            PreflightDiagnostic(
                "warn",
                "sandbox_resources",
                "sandbox resource snapshot unavailable",
                detail=f"{exc.__class__.__name__}: {str(exc)[:300]}",
            )
        ]
    if snapshot is None:
        return []

    capacity = estimate_run_capacity(plan)
    detail = _json_diagnostic_detail(
        {
            **snapshot.to_summary(),
            "max_active_sandboxes": capacity.max_active_sandboxes,
            "max_active_cpu_request": capacity.max_active_cpu_request,
            "max_active_memory_request_mb": capacity.max_active_memory_request_mb,
        }
    )
    if snapshot.node_count <= 0:
        return [
            PreflightDiagnostic(
                "warn",
                "sandbox_resources",
                f"sandbox provider returned no resource nodes: {snapshot.provider_name}",
                detail=detail,
            )
        ]
    if snapshot.ready_node_count <= 0:
        return [
            PreflightDiagnostic(
                "warn",
                "sandbox_resources",
                (f"sandbox resources have no ready nodes: 0/{snapshot.node_count} from {snapshot.provider_name}"),
                detail=detail,
            )
        ]
    if (snapshot.cpu_allocatable_m and capacity.max_active_cpu_request > snapshot.cpu_allocatable_m) or (
        snapshot.memory_allocatable_mb and capacity.max_active_memory_request_mb > snapshot.memory_allocatable_mb
    ):
        return [
            PreflightDiagnostic(
                "warn",
                "sandbox_resources",
                "sandbox resources may be insufficient for peak run demand",
                detail=detail,
            )
        ]
    return [
        PreflightDiagnostic(
            "ok",
            "sandbox_resources",
            (
                f"sandbox resources ready: {snapshot.ready_node_count}/"
                f"{snapshot.node_count} nodes from {snapshot.provider_name}"
            ),
            detail=detail,
        )
    ]


def _json_diagnostic_detail(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
