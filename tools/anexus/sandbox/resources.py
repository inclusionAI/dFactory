"""Sandbox provider resource snapshots."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SandboxResourceNode:
    """Normalized resource view for one sandbox cluster node."""

    node_id: str
    status: int = 0
    cpu_capacity_m: float = 0.0
    cpu_allocatable_m: float = 0.0
    memory_capacity_mb: float = 0.0
    memory_allocatable_mb: float = 0.0
    labels: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == 0

    @property
    def cpu_used_m(self) -> float:
        return max(self.cpu_capacity_m - self.cpu_allocatable_m, 0.0)

    @property
    def memory_used_mb(self) -> float:
        return max(self.memory_capacity_mb - self.memory_allocatable_mb, 0.0)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SandboxResourceSnapshot:
    """Normalized provider-level resource snapshot."""

    provider_name: str
    nodes: tuple[SandboxResourceNode, ...] = ()
    fetched_at: float = field(default_factory=time.time)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def ready_node_count(self) -> int:
        return sum(1 for node in self.nodes if node.ready)

    @property
    def cpu_capacity_m(self) -> float:
        return sum(node.cpu_capacity_m for node in self.nodes)

    @property
    def cpu_allocatable_m(self) -> float:
        return sum(node.cpu_allocatable_m for node in self.nodes)

    @property
    def cpu_used_m(self) -> float:
        return sum(node.cpu_used_m for node in self.nodes)

    @property
    def memory_capacity_mb(self) -> float:
        return sum(node.memory_capacity_mb for node in self.nodes)

    @property
    def memory_allocatable_mb(self) -> float:
        return sum(node.memory_allocatable_mb for node in self.nodes)

    @property
    def memory_used_mb(self) -> float:
        return sum(node.memory_used_mb for node in self.nodes)

    def to_summary(self) -> dict[str, object]:
        return {
            "provider_name": self.provider_name,
            "node_count": self.node_count,
            "ready_node_count": self.ready_node_count,
            "cpu_capacity_m": self.cpu_capacity_m,
            "cpu_allocatable_m": self.cpu_allocatable_m,
            "cpu_used_m": self.cpu_used_m,
            "memory_capacity_mb": self.memory_capacity_mb,
            "memory_allocatable_mb": self.memory_allocatable_mb,
            "memory_used_mb": self.memory_used_mb,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.to_summary(),
            "fetched_at": self.fetched_at,
            "nodes": [node.to_dict() for node in self.nodes],
        }


async def read_sandbox_provider_resource_snapshot(
    provider: object,
) -> SandboxResourceSnapshot | None:
    """Return a provider's normalized resource snapshot when it supports one."""

    snapshot_fn = getattr(provider, "resource_snapshot", None)
    if not callable(snapshot_fn):
        return None
    snapshot = await snapshot_fn()
    if isinstance(snapshot, SandboxResourceSnapshot):
        return snapshot
    raise TypeError("sandbox provider resource_snapshot() must return SandboxResourceSnapshot")


async def read_sandbox_provider_resource_snapshots(
    provider: object,
) -> tuple[SandboxResourceSnapshot, ...]:
    """Return one or more normalized resource snapshots when supported."""

    snapshots_fn = getattr(provider, "resource_snapshots", None)
    if callable(snapshots_fn):
        snapshots = await snapshots_fn()
        if not isinstance(snapshots, tuple):
            raise TypeError("sandbox provider resource_snapshots() must return tuple[SandboxResourceSnapshot, ...]")
        for snapshot in snapshots:
            if not isinstance(snapshot, SandboxResourceSnapshot):
                raise TypeError("sandbox provider resource_snapshots() must return SandboxResourceSnapshot items")
        return snapshots

    snapshot = await read_sandbox_provider_resource_snapshot(provider)
    return (snapshot,) if snapshot is not None else ()
