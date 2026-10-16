"""Swarm graph result reduction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from anexus.swarm.spec import SwarmPlan
from anexus.tasks import TaskAttemptRecord


@dataclass(frozen=True)
class SwarmEvaluationResult:
    """Graph-level result derived from node task-attempt records."""

    run_id: str
    swarm_id: str
    status: str
    passed: bool
    score: float | None
    node_count: int
    executed_node_count: int
    blocked_nodes: list[str] = field(default_factory=list)
    failed_nodes: list[str] = field(default_factory=list)
    degraded_nodes: list[str] = field(default_factory=list)
    status_counts: dict[str, int] = field(default_factory=dict)
    node_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable graph evaluation."""
        return {
            "run_id": self.run_id,
            "swarm_id": self.swarm_id,
            "status": self.status,
            "passed": self.passed,
            "score": self.score,
            "node_count": self.node_count,
            "executed_node_count": self.executed_node_count,
            "blocked_nodes": list(self.blocked_nodes),
            "failed_nodes": list(self.failed_nodes),
            "degraded_nodes": list(self.degraded_nodes),
            "status_counts": dict(self.status_counts),
            "node_scores": dict(self.node_scores),
        }


def reduce_swarm_result(
    plan: SwarmPlan,
    records: Mapping[str, TaskAttemptRecord],
    blocked_nodes: Iterable[str] = (),
) -> SwarmEvaluationResult:
    """Reduce node attempt records into one graph-level result."""
    ordered_nodes = _ordered_node_ids(plan)
    blocked = sorted(set(blocked_nodes))
    status_counts = Counter(records[node_id].status for node_id in ordered_nodes if node_id in records)
    if blocked:
        status_counts["blocked"] += len(blocked)

    status = _swarm_status(
        node_ids=ordered_nodes,
        records=records,
        blocked_nodes=blocked,
    )
    node_scores = _node_scores(ordered_nodes, records, blocked)
    return SwarmEvaluationResult(
        run_id=plan.swarm.run_id,
        swarm_id=plan.swarm.swarm_id,
        status=status,
        passed=status == "completed",
        score=_mean_node_score(ordered_nodes, node_scores),
        node_count=len(ordered_nodes),
        executed_node_count=sum(1 for node_id in ordered_nodes if node_id in records),
        blocked_nodes=blocked,
        failed_nodes=_failed_nodes(ordered_nodes, records, blocked),
        degraded_nodes=_degraded_nodes(ordered_nodes, records),
        status_counts=dict(status_counts),
        node_scores=node_scores,
    )


def _swarm_status(
    *,
    node_ids: list[str],
    records: Mapping[str, TaskAttemptRecord],
    blocked_nodes: list[str],
) -> str:
    if not node_ids:
        return "completed"
    if blocked_nodes:
        return "failed"
    if len(records) < len(node_ids):
        return "partial"

    statuses = {record.status for record in records.values()}
    if statuses == {"completed"}:
        return "completed"
    if "cancelled" in statuses:
        return "cancelled"
    if statuses <= {"completed", "cleanup_pending"}:
        return "cleanup_pending"
    if statuses <= {"completed", "artifact_pending"}:
        return "artifact_pending"
    if statuses == {"timeout"}:
        return "timeout"
    if statuses <= {"failed", "timeout"}:
        return "failed"
    if statuses & {"failed", "timeout"}:
        return "failed"
    return "partial"


def _node_scores(
    node_ids: list[str],
    records: Mapping[str, TaskAttemptRecord],
    blocked_nodes: list[str],
) -> dict[str, float]:
    blocked = set(blocked_nodes)
    scores: dict[str, float] = {}
    for node_id in node_ids:
        if node_id in blocked:
            scores[node_id] = 0.0
            continue
        record = records.get(node_id)
        if record is None:
            continue
        scores[node_id] = float(record.score or 0.0)
    return scores


def _mean_node_score(node_ids: list[str], node_scores: dict[str, float]) -> float | None:
    if not node_ids:
        return None
    return sum(node_scores.get(node_id, 0.0) for node_id in node_ids) / len(node_ids)


def _failed_nodes(
    node_ids: list[str],
    records: Mapping[str, TaskAttemptRecord],
    blocked_nodes: list[str],
) -> list[str]:
    failed = set(blocked_nodes)
    for node_id in node_ids:
        record = records.get(node_id)
        if record is not None and record.status in {"failed", "timeout", "cancelled"}:
            failed.add(node_id)
    return sorted(failed)


def _degraded_nodes(
    node_ids: list[str],
    records: Mapping[str, TaskAttemptRecord],
) -> list[str]:
    return [
        node_id
        for node_id in node_ids
        if records.get(node_id) is not None and records[node_id].status in {"artifact_pending", "cleanup_pending"}
    ]


def _ordered_node_ids(plan: SwarmPlan) -> list[str]:
    node_ids: list[str] = []
    for request in plan.node_requests:
        task = request.task
        node_id = task.labels.get("swarm_node_id")
        if not node_id:
            raise ValueError(f"task {task.task_id!r} does not have swarm_node_id label")
        node_ids.append(node_id)
    return node_ids
