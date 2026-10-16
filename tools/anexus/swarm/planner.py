"""Swarm graph validation and task compilation."""

from __future__ import annotations

from anexus.config import RunPlan, task_from_run_plan
from anexus.runtime.planning import ExecutionRequestCompiler
from anexus.swarm.spec import AgentEdgeSpec, AgentNodeSpec, SwarmPlan, SwarmTaskSpec


class SwarmPlanner:
    """Compile a swarm graph into strict node task specs."""

    def plan(
        self,
        swarm: SwarmTaskSpec,
        *,
        run_plan: RunPlan,
        compiler: ExecutionRequestCompiler | None = None,
    ) -> SwarmPlan:
        """Compile every swarm node from its resolved AgentSpec run plan.

        Args:
            swarm: Validated swarm graph input.
            run_plan: Multi-Agent run plan containing every node Agent.
            compiler: Optional compiler with caller-owned id generation.

        Returns:
            Dependency graph whose nodes contain complete execution requests.

        """
        node_ids = [node.node_id for node in swarm.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("swarm node_id values must be unique")

        task_identities = [(node.agent_name, node.task_id) for node in swarm.nodes]
        if len(set(task_identities)) != len(task_identities):
            raise ValueError("swarm (agent_name, task_id) values must be unique")

        known = set(node_ids)
        adjacency = {node_id: [] for node_id in node_ids}
        for edge in swarm.edges:
            if edge.source not in known:
                raise ValueError(f"edge source does not exist: {edge.source}")
            if edge.target not in known:
                raise ValueError(f"edge target does not exist: {edge.target}")
            adjacency[edge.source].append(edge.target)

        incoming_edges = _incoming_edges(swarm)
        request_compiler = compiler or ExecutionRequestCompiler()
        node_requests = []
        for node in swarm.nodes:
            _validate_node_run_plan(swarm, node, run_plan)
            task = task_from_run_plan(
                run_plan,
                {
                    **node.payload,
                    "task_id": node.task_id,
                    "agent_name": node.agent_name,
                },
                source_uri=f"{swarm.source_uri}#{node.node_id}",
                labels={
                    **swarm.labels,
                    **node.labels,
                    "swarm_id": swarm.swarm_id,
                    "swarm_node_id": node.node_id,
                    "gateway_scope": swarm.gateway_scope,
                    "gateway_session_group_id": _gateway_session_group_id(
                        swarm,
                        node,
                        incoming_edges[node.node_id],
                    ),
                },
            )
            node_requests.append(request_compiler.compile_task(run_plan, task))
        return SwarmPlan(swarm=swarm, node_requests=node_requests, adjacency=adjacency)


def _validate_node_run_plan(
    swarm: SwarmTaskSpec,
    node: AgentNodeSpec,
    plan: RunPlan,
) -> None:
    """Validate that one swarm node is present in the multi-Agent run plan."""
    if plan.run_id != swarm.run_id:
        raise ValueError(f"run plan {node.agent_spec_name!r} has a different run_id")
    agent = plan.agent(node.agent_name)
    if agent.agent_spec.name != node.agent_spec_name:
        raise ValueError(f"run plan key {node.agent_spec_name!r} does not match its AgentSpec")


def _incoming_edges(swarm: SwarmTaskSpec) -> dict[str, list[AgentEdgeSpec]]:
    incoming = {node.node_id: [] for node in swarm.nodes}
    for edge in swarm.edges:
        incoming[edge.target].append(edge)
    return incoming


def _gateway_session_group_id(
    swarm: SwarmTaskSpec,
    node: AgentNodeSpec,
    incoming_edges: list[AgentEdgeSpec],
) -> str:
    if swarm.gateway_scope == "run":
        return f"run:{swarm.run_id}"
    if swarm.gateway_scope == "swarm":
        return f"swarm:{swarm.swarm_id}"
    if swarm.gateway_scope == "node":
        return f"node:{swarm.swarm_id}:{node.node_id}"
    source_ids = "+".join(sorted(edge.source for edge in incoming_edges)) or "root"
    return f"edge:{swarm.swarm_id}:{source_ids}->{node.node_id}"
