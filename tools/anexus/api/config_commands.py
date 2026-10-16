"""Workspace, catalog-sync, and planning CLI commands."""

from __future__ import annotations

import asyncio
import json
from dataclasses import fields, is_dataclass
from hashlib import sha256
from typing import Any

from anexus.api.cli_context import (
    apply_workspace_environment,
    catalog_source_from_workspace,
    resolve_plan,
    restore_environment,
    workspace_from_input,
)
from anexus.api.command_inputs import InitCommandInput, RunCommandInput, UpdateCommandInput
from anexus.api.preflight import preflight_run_plan
from anexus.config import (
    CatalogCache,
    RunPlan,
    bootstrap_workspace,
)


def init_command(command: InitCommandInput) -> int:
    """Initialize the workspace without duplicating catalog configuration as flags."""

    result = bootstrap_workspace(
        workspace_from_input(command).root,
        force=command.force,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def plan_command(command: RunCommandInput) -> int:
    """Resolve and statically validate a plan without creating providers."""

    saved_env = apply_workspace_environment(command)
    try:
        plan = resolve_plan(command)
        report = asyncio.run(preflight_run_plan(plan, check_object_fixtures=False))
    finally:
        restore_environment(saved_env)
    print(
        json.dumps(
            {
                "run_id": plan.run_id,
                "agent_names": list(plan.agent_names),
                "task_count": len(plan.tasks),
                "tasks": [{"task_id": task.task_id, "agent_name": task.agent_name} for task in plan.tasks],
                "agents": {agent_name: _agent_plan_summary(plan, agent_name) for agent_name in plan.agent_names},
                "cluster": _to_jsonable(plan.cluster),
                "n_concurrent": plan.n_concurrent,
                "n_attempts": plan.n_attempts,
                "agent_concurrency_limits": dict(plan.agent_concurrency_limits),
                "agent_concurrency_groups": dict(plan.agent_concurrency_groups),
                "validation": report.to_dict(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if report.has_errors else 0


def update_command(command: UpdateCommandInput) -> int:
    """Synchronize the catalog selected in ``anexus.config.yaml``."""

    cache_root = workspace_from_input(command).cache_dir
    source = catalog_source_from_workspace(command)
    synced = source.sync(cache_root)
    result = CatalogCache(cache_root).update(synced)
    print(
        json.dumps(
            {
                "source_uri": synced.source_uri,
                "fetched_at": synced.fetched_at,
                "content_sha256": sha256(synced.content.encode()).hexdigest(),
                "repository": synced.repository,
                "branch": synced.branch,
                "path": synced.path,
                "commit_id": synced.commit_id,
                "resolved_commit_id": synced.resolved_commit_id,
                "version": result.catalog.version,
                "agent_count": len(result.catalog.agents),
                "agents": result.catalog.names(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _to_jsonable(value: object) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):  # type: ignore[attr-defined]
        return _to_jsonable(value.to_dict())  # type: ignore[attr-defined]
    if is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    return value


def _task_policy_summary(plan: RunPlan, agent_name: str) -> dict[str, Any]:
    tasks = [task for task in plan.tasks if task.agent_name == agent_name]
    if not tasks:
        return {}
    policy = tasks[0].attempt_policy
    return {
        "timeout_seconds": policy.timeout_seconds,
        "max_retries": policy.max_retries,
        "min_retry_wait_seconds": policy.min_retry_wait_seconds,
        "max_retry_wait_seconds": policy.max_retry_wait_seconds,
        "retry_wait_multiplier": policy.retry_wait_multiplier,
    }


def _agent_plan_summary(plan: RunPlan, agent_name: str) -> dict[str, Any]:
    agent = plan.agent(agent_name)
    return {
        "agent_spec": agent.agent_spec.name,
        "task_count": sum(task.agent_name == agent_name for task in plan.tasks),
        "sandbox": {
            "provider": agent.sandbox.provider,
            "image": agent.sandbox.image,
            "cpu_request": agent.sandbox.cpu_request,
            "cpu_limit": agent.sandbox.cpu_limit,
            "memory_request_mb": agent.sandbox.memory_request_mb,
            "memory_limit_mb": agent.sandbox.memory_limit_mb,
        },
        "sidecars": [_to_jsonable(sidecar) for sidecar in agent.sidecars],
        "gateway": {
            "name": agent.gateway.name,
            "driver": agent.gateway.driver,
            "sandbox_provider": agent.gateway.sandbox_provider,
            "resources": _to_jsonable(agent.gateway.resources),
            "mirrors": [_mirror_summary(mirror) for mirror in agent.gateway.mirrors],
            "agent_models": dict(agent.agent_models),
            "judge_model_id": agent.judge_model_id,
        },
        "agent_runtime": {
            "name": agent.agent_runtime.name,
            "driver": agent.agent_runtime.driver,
            "protocol": agent.agent_runtime.protocol,
        },
        "task_policy": _task_policy_summary(plan, agent_name),
    }


def _mirror_summary(mirror: object) -> dict[str, Any]:
    return {
        "name": getattr(mirror, "name", ""),
        "models": list(getattr(mirror, "models", ())),
        "openai_upstream": getattr(mirror, "openai_upstream", ""),
        "anthropic_upstream": getattr(mirror, "anthropic_upstream", ""),
    }
