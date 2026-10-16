"""Retryable cleanup backlog for leaked runtime resources."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from anexus.core.context import TaskContext
from anexus.core.errors import ErrorEnvelope
from anexus.runtime.store import safe_path_part
from anexus.sandbox import SandboxLease

CleanupStatus = Literal["pending"]


@dataclass(frozen=True)
class CleanupEntry:
    name: str
    status: CleanupStatus
    resource_kind: Literal["sandbox"]
    resource_id: str
    provider: str
    context: dict[str, Any]
    error: dict[str, Any] | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    resource_name: str = ""
    resource_role: str = ""

    @classmethod
    def sandbox_pending(
        cls,
        *,
        context: TaskContext,
        lease: SandboxLease,
        error: ErrorEnvelope,
        resource_name: str = "",
        resource_role: str = "",
    ) -> CleanupEntry:
        return cls(
            name="sandbox.cleanup_pending",
            status="pending",
            resource_kind="sandbox",
            resource_id=lease.sandbox_id,
            provider=lease.provider,
            context=context.to_fields(),
            error=error.to_dict(),
            resource_name=resource_name,
            resource_role=resource_role,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CleanupBacklog:
    """Append-only cleanup backlog scoped to one run."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = Lock()

    def path(self, run_id: str) -> Path:
        return self.root / "runs" / safe_path_part(run_id) / "cleanup.jsonl"

    def add(self, run_id: str, entry: CleanupEntry) -> None:
        encoded = json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True)
        path = self.path(run_id)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.write("\n")
