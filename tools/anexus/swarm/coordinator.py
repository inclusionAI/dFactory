"""Swarm graph coordinator."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from anexus.core.execution import ExecutionRequest
from anexus.events import EventBus, EventEnvelope
from anexus.runtime.queue import AttemptRunner, QueuePolicy, TaskRunQueue
from anexus.swarm.reducer import SwarmEvaluationResult, reduce_swarm_result
from anexus.swarm.spec import AgentEdgeSpec, SwarmPlan
from anexus.tasks import TaskAttemptRecord


@dataclass(frozen=True)
class SwarmPolicy:
    """Define global and per-agent concurrency for a swarm."""

    n_concurrent: int = 10
    agent_concurrency_limits: dict[str, int] = field(default_factory=dict)
    agent_concurrency_groups: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate swarm concurrency through the canonical queue policy."""
        if self.n_concurrent < 1:
            raise ValueError("SwarmPolicy.n_concurrent must be >= 1")
        QueuePolicy(
            n_concurrent=self.n_concurrent,
            agent_concurrency_limits=self.agent_concurrency_limits,
            agent_concurrency_groups=self.agent_concurrency_groups,
        )


@dataclass(frozen=True)
class SwarmResult:
    """Aggregate node records and graph-level evaluation."""

    run_id: str
    swarm_id: str
    status: str
    node_records: dict[str, TaskAttemptRecord] = field(default_factory=dict)
    blocked_nodes: list[str] = field(default_factory=list)
    evaluation: SwarmEvaluationResult | None = None
    score: float | None = None
    status_counts: dict[str, int] = field(default_factory=dict)


class SwarmCoordinator:
    """Execute a compiled swarm graph by dependency wave."""

    def __init__(self, *, bus: EventBus, policy: SwarmPolicy | None = None):
        """Initialize the coordinator with an event bus and queue policy."""
        self.bus = bus
        self.policy = policy or SwarmPolicy()

    def run(self, plan: SwarmPlan, *, runner: AttemptRunner) -> SwarmResult:
        """Execute a swarm plan in a new event loop.

        Args:
            plan: Compiled swarm requests and dependency graph.
            runner: Run-scoped canonical attempt runner.

        Returns:
            Aggregated swarm result.

        """
        return asyncio.run(self.run_async(plan, runner=runner))

    async def run_async(self, plan: SwarmPlan, *, runner: AttemptRunner) -> SwarmResult:
        """Execute a swarm graph dependency wave by dependency wave.

        Args:
            plan: Compiled swarm requests and dependency graph.
            runner: Run-scoped canonical attempt runner.

        Returns:
            Aggregated swarm result.

        """
        self.log("swarm.started", plan, node_count=len(plan.node_requests))
        records: dict[str, TaskAttemptRecord] = {}
        blocked: set[str] = set()
        pending = {_node_id(request) for request in plan.node_requests}
        request_by_node = {_node_id(request): request for request in plan.node_requests}
        incoming = _incoming_edges(plan)

        while pending:
            newly_blocked = _blocked_nodes(pending, incoming, records, blocked)
            for node_id in newly_blocked:
                pending.remove(node_id)
                blocked.add(node_id)
                self.log("swarm.node.blocked", plan, node_id=node_id)

            ready = [
                node_id
                for node_id in _ordered_nodes(plan.node_requests)
                if node_id in pending and _node_ready(node_id, incoming, records, blocked)
            ]
            if not ready:
                if pending:
                    raise ValueError(f"swarm graph has unresolved dependencies: {sorted(pending)}")
                break

            self.log("swarm.wave.started", plan, node_ids=ready)
            queue = TaskRunQueue(
                policy=QueuePolicy(
                    n_concurrent=self.policy.n_concurrent,
                    agent_concurrency_limits=self.policy.agent_concurrency_limits,
                    agent_concurrency_groups=self.policy.agent_concurrency_groups,
                ),
                runner=runner,
                event_publisher=self.bus.publish,
            )
            wave_records = await queue.run([request_by_node[node_id] for node_id in ready])
            for node_id, record in zip(ready, wave_records, strict=True):
                records[node_id] = record
                pending.remove(node_id)
                self.log(
                    "swarm.node.completed",
                    plan,
                    node_id=node_id,
                    status=record.status,
                    score=record.score,
                )

        evaluation = reduce_swarm_result(plan, records, blocked)
        result = SwarmResult(
            run_id=plan.swarm.run_id,
            swarm_id=plan.swarm.swarm_id,
            status=evaluation.status,
            node_records=records,
            blocked_nodes=evaluation.blocked_nodes,
            evaluation=evaluation,
            score=evaluation.score,
            status_counts=evaluation.status_counts,
        )
        self.log(
            "swarm.completed",
            plan,
            status=result.status,
            node_count=evaluation.node_count,
            executed_node_count=evaluation.executed_node_count,
            score=evaluation.score,
            blocked_nodes=result.blocked_nodes,
            failed_nodes=evaluation.failed_nodes,
            degraded_nodes=evaluation.degraded_nodes,
            status_counts=evaluation.status_counts,
        )
        return result

    def log(self, name: str, plan: SwarmPlan, **fields: object) -> None:
        """Record one swarm event at the observability boundary."""
        self.bus.publish(
            EventEnvelope(
                name=name,
                context={
                    "run_id": plan.swarm.run_id,
                    "swarm_id": plan.swarm.swarm_id,
                    "gateway_scope": plan.swarm.gateway_scope,
                },
                payload=fields,
            )
        )


def _incoming_edges(plan: SwarmPlan) -> dict[str, list[AgentEdgeSpec]]:
    incoming = {_node_id(request): [] for request in plan.node_requests}
    for edge in plan.swarm.edges:
        incoming[edge.target].append(edge)
    return incoming


def _ordered_nodes(requests: list[ExecutionRequest]) -> list[str]:
    return [_node_id(request) for request in requests]


def _node_ready(
    node_id: str,
    incoming: dict[str, list[AgentEdgeSpec]],
    records: dict[str, TaskAttemptRecord],
    blocked: set[str],
) -> bool:
    for edge in incoming[node_id]:
        if edge.source not in records and edge.source not in blocked:
            return False
        if edge.required and edge.source in blocked:
            return False
        if edge.required and records.get(edge.source) is not None:
            if records[edge.source].status != "completed":
                return False
    return True


def _blocked_nodes(
    pending: set[str],
    incoming: dict[str, list[AgentEdgeSpec]],
    records: dict[str, TaskAttemptRecord],
    blocked: set[str],
) -> list[str]:
    blocked_nodes: list[str] = []
    for node_id in sorted(pending):
        for edge in incoming[node_id]:
            if not edge.required:
                continue
            if edge.source in blocked:
                blocked_nodes.append(node_id)
                break
            source_record = records.get(edge.source)
            if source_record is not None and source_record.status != "completed":
                blocked_nodes.append(node_id)
                break
    return blocked_nodes


def _node_id(request: ExecutionRequest) -> str:
    task = request.task
    node_id = task.labels.get("swarm_node_id")
    if not node_id:
        raise ValueError(f"task {task.task_id!r} does not have swarm_node_id label")
    return node_id
