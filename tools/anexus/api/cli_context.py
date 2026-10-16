"""Workspace, environment, and plan resolution shared by CLI commands."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

from anexus.api.command_inputs import CommandInput, RunCommandInput
from anexus.config import (
    CatalogCache,
    CatalogSource,
    RunOverrides,
    RunPlan,
    load_config_data,
    load_task_payloads,
    make_workspace,
    parse_config_text,
    resolve_catalog_for_agents,
    resolve_run_plan,
    resolve_workspace,
    resolve_workspace_environment,
)
from anexus.core.ids import new_id


def resolve_plan(command: RunCommandInput) -> RunPlan:
    """Resolve the CLI's small run input into a complete provider-free plan."""

    workspace = workspace_from_input(command)
    workspace_env, _workspace_config = resolve_workspace_environment(workspace)
    source = catalog_source_from_workspace(command)
    cached = CatalogCache(workspace.cache_dir).load(source)
    if cached is None:
        raise ValueError("no local catalog checkout found; run anexus update first")
    catalog_uri = cached.source_uri
    task_uri = uri_from_workspace(command.tasks, command)
    task_path = Path(unquote(urlsplit(task_uri).path))
    task_ids = set(command.task_ids) or None
    task_payloads = load_task_payloads(task_path, task_ids=task_ids)
    agent_names = {
        str(payload.get("agent_name", "")).strip()
        for payload in task_payloads
        if str(payload.get("agent_name", "")).strip()
    }
    raw_catalog_data = parse_config_text(cached.content, source_uri=catalog_uri)
    catalog = resolve_catalog_for_agents(
        raw_catalog_data,
        agent_names=agent_names,
        env=workspace_env,
        source_uri=catalog_uri,
        task_payloads=task_payloads,
    )
    return resolve_run_plan(
        catalog=catalog,
        run_id=command.run_id or new_id("run"),
        catalog_uri=catalog_uri,
        task_payloads=task_payloads,
        task_source_uri=task_uri,
        source_uri=task_uri,
        overrides=RunOverrides(
            task_ids=task_ids,
            n_concurrent=command.concurrency,
            n_attempts=command.attempts,
        ),
    )


def catalog_source_from_workspace(command: CommandInput) -> CatalogSource:
    """Read the remote catalog locator from workspace configuration."""

    workspace = workspace_from_input(command)
    data = load_config_data(workspace.config_path) if workspace.config_path.exists() else {}
    return CatalogSource.from_workspace_config(data)


def workspace_from_input(command: CommandInput):
    """Resolve a typed command's workspace."""

    return make_workspace(resolve_workspace(command.workspace or None))


def apply_workspace_environment(command: CommandInput) -> dict[str, str | None]:
    """Apply the resolved workspace environment and return replaced values."""

    resolved, _config = resolve_workspace_environment(workspace_from_input(command))
    old: dict[str, str | None] = {}
    for key, value in resolved.items():
        old.setdefault(key, os.environ.get(key))
        os.environ[key] = value
    return old


def restore_environment(old: dict[str, str | None]) -> None:
    """Restore environment values replaced for one command invocation."""

    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def path_from_workspace(path: str | Path, command: CommandInput) -> Path:
    """Resolve a local path from the command workspace."""

    local_path = Path(path).expanduser()
    if local_path.is_absolute():
        return local_path
    return workspace_from_input(command).root / local_path


def uri_from_workspace(uri: str, command: CommandInput) -> str:
    """Resolve the CLI's intentionally local task input."""

    parts = urlsplit(uri)
    if parts.scheme == "file":
        return Path(unquote(parts.path)).expanduser().resolve().as_uri()
    if parts.scheme:
        raise ValueError("task JSONL must be a local file")
    return path_from_workspace(uri, command).resolve().as_uri()
