"""Pure run-capacity estimates for scheduling and preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from anexus.config import RunPlan


@dataclass(frozen=True)
class RunCapacity:
    """Resource and attempt counts implied by a resolved run plan."""

    task_count: int
    attempts_per_task: int
    total_attempts: int
    agent_concurrency_limit: int
    max_active_agent_phases: int
    max_active_attempts: int
    sidecar_count: int
    verifier_sandbox_count: int
    sandboxes_per_attempt: int
    total_sandbox_creates: int
    max_active_sandboxes: int
    cpu_request_per_attempt: int
    cpu_limit_per_attempt: int
    memory_request_mb_per_attempt: int
    memory_limit_mb_per_attempt: int
    max_active_cpu_request: int
    max_active_cpu_limit: int
    max_active_memory_request_mb: int
    max_active_memory_limit_mb: int
    gateway_cpu_request: int
    gateway_cpu_limit: int
    gateway_memory_request: int
    gateway_memory_limit: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def estimate_run_capacity(plan: RunPlan) -> RunCapacity:
    """Estimate scheduling and sandbox capacity without creating resources."""

    task_count = len(plan.tasks)
    total_attempts = task_count * plan.n_attempts
    max_active_attempts = min(plan.n_concurrent, total_attempts) if total_attempts else 0
    attempts_by_pool: dict[str, int] = {}
    for task in plan.tasks:
        pool_name = plan.agent_concurrency_groups.get(task.agent_name, task.agent_name)
        attempts_by_pool[pool_name] = attempts_by_pool.get(pool_name, 0) + plan.n_attempts
    configured_limits = {
        pool_name: plan.agent_concurrency_limits[pool_name]
        for pool_name in attempts_by_pool
        if pool_name in plan.agent_concurrency_limits
    }
    agent_concurrency_limit = sum(configured_limits.values())
    pool_capacity = sum(
        min(attempt_count, configured_limits.get(pool_name, attempt_count))
        for pool_name, attempt_count in attempts_by_pool.items()
    )
    max_active_agent_phases = min(
        max_active_attempts,
        pool_capacity,
    )
    sidecar_count = max((len(agent.sidecars) for agent in plan.agents.values()), default=0)
    verifier_sandbox_count = max(
        (_isolated_verifier_sandbox_count(agent.sandbox.verify) for agent in plan.agents.values()),
        default=0,
    )
    sandboxes_per_attempt = 1 + sidecar_count + verifier_sandbox_count
    total_sandbox_creates = sum(
        sum(1 for task in plan.tasks if task.agent_name == name)
        * plan.n_attempts
        * (1 + len(agent.sidecars) + _isolated_verifier_sandbox_count(agent.sandbox.verify))
        for name, agent in plan.agents.items()
    )
    max_active_sandboxes = max_active_attempts * sandboxes_per_attempt

    cpu_request_per_attempt = max(
        (
            agent.sandbox.cpu_request + sum(sidecar.cpu_request for sidecar in agent.sidecars)
            for agent in plan.agents.values()
        ),
        default=0,
    )
    cpu_limit_per_attempt = max(
        (
            agent.sandbox.cpu_limit + sum(sidecar.cpu_limit for sidecar in agent.sidecars)
            for agent in plan.agents.values()
        ),
        default=0,
    )
    memory_request_mb_per_attempt = max(
        (
            agent.sandbox.memory_request_mb + sum(sidecar.memory_request_mb for sidecar in agent.sidecars)
            for agent in plan.agents.values()
        ),
        default=0,
    )
    memory_limit_mb_per_attempt = max(
        (
            agent.sandbox.memory_limit_mb + sum(sidecar.memory_limit_mb for sidecar in agent.sidecars)
            for agent in plan.agents.values()
        ),
        default=0,
    )

    return RunCapacity(
        task_count=task_count,
        attempts_per_task=plan.n_attempts,
        total_attempts=total_attempts,
        agent_concurrency_limit=agent_concurrency_limit,
        max_active_agent_phases=max_active_agent_phases,
        max_active_attempts=max_active_attempts,
        sidecar_count=sidecar_count,
        verifier_sandbox_count=verifier_sandbox_count,
        sandboxes_per_attempt=sandboxes_per_attempt,
        total_sandbox_creates=total_sandbox_creates,
        max_active_sandboxes=max_active_sandboxes,
        cpu_request_per_attempt=cpu_request_per_attempt,
        cpu_limit_per_attempt=cpu_limit_per_attempt,
        memory_request_mb_per_attempt=memory_request_mb_per_attempt,
        memory_limit_mb_per_attempt=memory_limit_mb_per_attempt,
        max_active_cpu_request=cpu_request_per_attempt * max_active_attempts,
        max_active_cpu_limit=cpu_limit_per_attempt * max_active_attempts,
        max_active_memory_request_mb=memory_request_mb_per_attempt * max_active_attempts,
        max_active_memory_limit_mb=memory_limit_mb_per_attempt * max_active_attempts,
        gateway_cpu_request=sum(agent.gateway.resources.cpu_request for agent in plan.agents.values()),
        gateway_cpu_limit=sum(agent.gateway.resources.cpu_limit for agent in plan.agents.values()),
        gateway_memory_request=sum(agent.gateway.resources.memory_request for agent in plan.agents.values()),
        gateway_memory_limit=sum(agent.gateway.resources.memory_limit for agent in plan.agents.values()),
    )


def _isolated_verifier_sandbox_count(verify: dict[str, object] | None) -> int:
    if not isinstance(verify, dict):
        return 0
    if verify.get("enabled", True) is False:
        return 0
    sandbox = verify.get("sandbox")
    if "grader" in verify and "isolated" not in verify and not (isinstance(sandbox, dict) and sandbox):
        return 1
    if bool(verify.get("isolated", False)):
        return 1
    if isinstance(sandbox, dict) and sandbox:
        return 1
    return 0
