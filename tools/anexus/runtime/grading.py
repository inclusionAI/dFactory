"""Provider-backed grader helpers."""

from __future__ import annotations

import tempfile
from hashlib import sha256
from pathlib import Path, PurePosixPath

from anexus.sandbox.provider import ExecResult, SandboxLease, SandboxProvider
from anexus.tasks.graders import GraderConfig, get_grader


def get_grader_or_raise(config: GraderConfig):
    """Resolve a configured grader."""

    return get_grader(config.grader)


class ProviderArtifactSandbox:
    """Expose the grader artifact protocol over a provider lease."""

    def __init__(
        self,
        *,
        sandbox_provider: SandboxProvider,
        lease: SandboxLease,
    ) -> None:
        self._sandbox_provider = sandbox_provider
        self._lease = lease
        self.sandbox_id = lease.sandbox_id
        self.instance_id = lease.sandbox_id

    async def read_file(self, path: str) -> str:
        target = Path(tempfile.mkdtemp()) / _artifact_filename(path)
        await self._sandbox_provider.get_file(self._lease, path, target)
        return target.read_text(encoding="utf-8")

    async def run_command(self, command: str, *, timeout: int = 60) -> ExecResult:
        return await self._sandbox_provider.exec(self._lease, command, timeout=timeout)


def _artifact_filename(path: str) -> str:
    name = PurePosixPath(path).name or "artifact"
    digest = sha256(path.encode("utf-8")).hexdigest()[:16]
    return f"{digest}-{name}"
