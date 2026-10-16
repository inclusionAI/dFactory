"""Agent swarm graph specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from anexus.core.execution import ExecutionRequest


@dataclass(frozen=True)
class AgentNodeSpec:
    """Describe one agent node and its task payload."""

    node_id: str
    task_id: str
    agent_name: str
    payload: dict[str, Any]
    agent_spec_name: str
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentEdgeSpec:
    """Describe one directed dependency between swarm nodes."""

    source: str
    target: str
    required: bool = True


@dataclass(frozen=True)
class SwarmTaskSpec:
    """Define a swarm graph before execution requests are compiled."""

    run_id: str
    swarm_id: str
    source_uri: str
    nodes: list[AgentNodeSpec]
    edges: list[AgentEdgeSpec] = field(default_factory=list)
    gateway_scope: Literal["run", "swarm", "node", "edge"] = "swarm"
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the gateway sharing scope."""
        if self.gateway_scope not in {"run", "swarm", "node", "edge"}:
            raise ValueError("SwarmTaskSpec.gateway_scope must be run, swarm, node, or edge")


@dataclass(frozen=True)
class SwarmPlan:
    """Contain compiled node requests and graph adjacency."""

    swarm: SwarmTaskSpec
    node_requests: list[ExecutionRequest]
    adjacency: dict[str, list[str]]
