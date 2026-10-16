"""Workspace path discovery and full workspace resolution."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

WORKSPACE_ENV_VAR = "ANEXUS_WORKSPACE"
CONFIG_FILE_NAME = "anexus.config.yaml"
DOTENV_FILE_NAME = ".env"
CACHE_DIR_NAME = ".anexus"
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Workspace:
    """Resolved workspace paths for local CLI and cache operations."""

    root: Path
    config_path: Path
    dotenv_path: Path
    cache_dir: Path


def make_workspace(root: str | Path) -> Workspace:
    """Build workspace paths from a root directory."""

    root_path = Path(root).expanduser().resolve()
    cache_dir = root_path / CACHE_DIR_NAME
    return Workspace(
        root=root_path,
        config_path=root_path / CONFIG_FILE_NAME,
        dotenv_path=root_path / DOTENV_FILE_NAME,
        cache_dir=cache_dir,
    )


def resolve_workspace(
    explicit: str | Path | None = None,
    *,
    strict: bool = False,
    cwd: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the workspace root without walking parent directories."""

    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    env = os.environ if environ is None else environ
    env_workspace = env.get(WORKSPACE_ENV_VAR)
    if env_workspace:
        return Path(env_workspace).expanduser().resolve()

    cwd_path = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd().resolve()
    if _looks_like_workspace(cwd_path):
        return cwd_path
    if strict:
        raise RuntimeError(
            "No anexus workspace found. Pass --workspace, set ANEXUS_WORKSPACE, "
            "or run from a directory containing anexus.config.yaml or .anexus."
        )
    return cwd_path


def _looks_like_workspace(path: Path) -> bool:
    return (path / CONFIG_FILE_NAME).is_file() or (path / CACHE_DIR_NAME).is_dir()


def resolve_workspace_environment(
    workspace: Workspace,
    *,
    env_overrides: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    """Resolve workspace config and environment using the CLI precedence order."""

    from anexus.config.loader import load_config_data
    from anexus.config.settings import load_env_file

    dotenv_env = load_env_file(workspace.dotenv_path)
    host_env = dict(os.environ if environ is None else environ)
    reference_env = {**dotenv_env, **host_env, **(env_overrides or {})}
    config: dict[str, object] = {}
    workspace_env: dict[str, str] = {}
    if workspace.config_path.exists():
        config = load_config_data(workspace.config_path)
        workspace_env = _workspace_config_env(
            config,
            source_uri=workspace.config_path.resolve().as_uri(),
            reference_env=reference_env,
        )

    merged_env = {
        **workspace_env,
        **dotenv_env,
        **host_env,
        **(env_overrides or {}),
    }
    return merged_env, config


def _workspace_config_env(
    config: Mapping[str, object],
    *,
    source_uri: str,
    reference_env: Mapping[str, str],
) -> dict[str, str]:
    """Resolve the canonical top-level workspace ``env`` overlay."""
    from anexus.config.environment import env_refs_resolvable, resolve_env_refs
    from anexus.config.loader import ConfigLoadError

    raw_env = config.get("env")
    if raw_env is None:
        return {}
    if not isinstance(raw_env, Mapping):
        raise ConfigLoadError(source_uri, "env must be an object")

    resolved_env: dict[str, str] = {}
    for raw_name, value in raw_env.items():
        name = str(raw_name)
        if _ENV_NAME_RE.fullmatch(name) is None:
            raise ConfigLoadError(source_uri, f"env contains invalid variable name {name!r}")
        if isinstance(value, (Mapping, list, tuple)) or value is None:
            raise ConfigLoadError(source_uri, f"env.{name} must be a scalar value")
        if not env_refs_resolvable(value, env=reference_env):
            continue
        resolved = resolve_env_refs(
            {name: value},
            env=reference_env,
            source_uri=source_uri,
        )
        assert isinstance(resolved, dict)
        resolved_env[name] = str(resolved[name])
    return resolved_env
