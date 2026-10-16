"""Workspace bootstrap files for local CLI use."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from anexus.config.workspace import Workspace, make_workspace

BootstrapStatus = Literal["created", "overwritten", "skipped"]


@dataclass(frozen=True)
class BootstrapFile:
    """Record the result of writing one bootstrap file."""

    path: Path
    status: BootstrapStatus

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable file result."""
        return {"path": str(self.path), "status": self.status}


@dataclass(frozen=True)
class WorkspaceBootstrapResult:
    """Collect all files created for a workspace bootstrap."""

    workspace: Workspace
    files: list[BootstrapFile]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable bootstrap result."""
        return {
            "workspace": str(self.workspace.root),
            "files": [item.to_dict() for item in self.files],
        }


def bootstrap_workspace(
    root: str | Path,
    *,
    catalog_repository: str = "",
    catalog_branch: str = "",
    catalog_path: str = "",
    catalog_commit_id: str = "",
    force: bool = False,
) -> WorkspaceBootstrapResult:
    """Create starter workspace files without touching runtime resources."""
    workspace = make_workspace(root)
    workspace.root.mkdir(parents=True, exist_ok=True)
    files = [
        _write_text(
            workspace.config_path,
            _workspace_config_text(
                catalog_repository=catalog_repository,
                catalog_branch=catalog_branch,
                catalog_path=catalog_path,
                catalog_commit_id=catalog_commit_id,
            ),
            force=force,
        ),
        _write_text(workspace.root / ".env.example", _env_example_text(), force=force),
    ]
    workspace.cache_dir.mkdir(parents=True, exist_ok=True)
    return WorkspaceBootstrapResult(workspace=workspace, files=files)


def _write_text(path: Path, content: str, *, force: bool) -> BootstrapFile:
    path = path.expanduser()
    if path.exists() and not force:
        return BootstrapFile(path=path, status="skipped")
    path.parent.mkdir(parents=True, exist_ok=True)
    status: BootstrapStatus = "overwritten" if path.exists() else "created"
    path.write_text(content, encoding="utf-8")
    return BootstrapFile(path=path, status=status)


def _workspace_config_text(
    *,
    catalog_repository: str,
    catalog_branch: str,
    catalog_path: str,
    catalog_commit_id: str,
) -> str:
    return f"""# Anexus workspace configuration.
#
# Keep credentials in .env or the process environment. Runtime resources are
# created only by run or explicit integration checks, never by init/update/plan/inspect.

catalog:
  repository: {json.dumps(catalog_repository)}
  branch: {json.dumps(catalog_branch)}
  path: {json.dumps(catalog_path)}
  commit_id: {json.dumps(catalog_commit_id)}

# Workspace-local values for `${{env:NAME}}` placeholders in the catalog.
# Replace a right-hand side with a literal to override it here. Keeping the
# reference delegates the value to .env or the process environment.
env:
  AKERNEL_INTERNAL_ADDRESS: "${{env:AKERNEL_INTERNAL_ADDRESS}}"
  AKERNEL_TOKEN: "${{env:AKERNEL_TOKEN}}"
  OSS_KEY_ID: "${{env:OSS_KEY_ID}}"
  OSS_KEY_SECRET: "${{env:OSS_KEY_SECRET}}"
  MIRROR_API_KEY: "${{env:MIRROR_API_KEY}}"
  MIRROR_OPENAI_UPSTREAM: "${{env:MIRROR_OPENAI_UPSTREAM}}"
  MIRROR_ANTHROPIC_UPSTREAM: "${{env:MIRROR_ANTHROPIC_UPSTREAM}}"
  TRACE_VIEWER_URL: "${{env:TRACE_VIEWER_URL}}"
"""


def _env_example_text() -> str:
    return """# Copy this file to .env and fill local credentials.

# Gateway upstream.
MIRROR_API_KEY=
MIRROR_OPENAI_UPSTREAM=
MIRROR_ANTHROPIC_UPSTREAM=

# Sandbox provider.
AKERNEL_INTERNAL_ADDRESS=
AKERNEL_TOKEN=

# Object store.
OSS_KEY_ID=
OSS_KEY_SECRET=

# Trace viewer override (optional).
TRACE_VIEWER_URL=
"""
