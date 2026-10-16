"""Thin reward-return adapter for external RL rollout systems."""

from __future__ import annotations

from typing import Any, Protocol

from anexus.config import RunPlan, task_from_run_plan
from anexus.rl.spec import RewardSpec, RolloutOutcome, RolloutTaskSpec
from anexus.runtime.planning import ExecutionRequestCompiler


class RolloutRunner(Protocol):
    """Minimal runner contract used by external RL callback adapters."""

    async def run(self, rollout: RolloutTaskSpec) -> RolloutOutcome:
        """Execute one compiled rollout and return its reward outcome."""
        ...


def build_rollout_task(
    data: dict[str, Any],
    *,
    plan: RunPlan,
    compiler: ExecutionRequestCompiler | None = None,
    source_uri: str = "memory://rollout",
    episode_id: str = "",
    reward_spec: RewardSpec | None = None,
) -> RolloutTaskSpec:
    """Compile one RL payload through the same task and request path as a run.

    Args:
        data: Raw RL callback payload.
        plan: Resolved catalog and run decisions.
        compiler: Optional request compiler with caller-owned id generation.
        source_uri: Source identifier recorded on the task.
        episode_id: Optional external episode identifier.
        reward_spec: Reward extraction policy.

    Returns:
        A rollout containing a complete execution request.

    """
    payload = dict(data)
    if "task_id" not in payload and payload.get("instance_id"):
        payload["task_id"] = str(payload["instance_id"])

    task = task_from_run_plan(
        plan,
        payload,
        source_uri=source_uri,
        labels={"profile": "rl"},
    )
    rollout_episode_id = episode_id or str(payload.get("episode_id", "")) or task.task_id
    metadata = {
        key: value for key, value in payload.items() if key not in {"prompt", "task_id", "instance_id", "agent_name"}
    }
    return RolloutTaskSpec(
        request=(compiler or ExecutionRequestCompiler()).compile_task(plan, task),
        episode_id=rollout_episode_id,
        reward_spec=reward_spec or RewardSpec(),
        rollout_metadata=metadata,
    )


async def run_agent_return_reward(
    data: dict[str, Any],
    *,
    runner: RolloutRunner,
    plan: RunPlan,
    compiler: ExecutionRequestCompiler | None = None,
    source_uri: str = "memory://rollout",
    episode_id: str = "",
    reward_spec: RewardSpec | None = None,
    raise_on_failed: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Run one compiled rollout and return its reward and summary fields."""
    rollout = build_rollout_task(
        data,
        plan=plan,
        compiler=compiler,
        source_uri=source_uri,
        episode_id=episode_id,
        reward_spec=reward_spec,
    )
    outcome = await runner.run(rollout)
    if raise_on_failed and outcome.attempt.status in {"failed", "timeout", "cancelled"}:
        raise RuntimeError(
            f"rollout failed for {outcome.attempt.agent_name}/{outcome.attempt.task_id}: {outcome.attempt.status}"
        )
    return outcome.reward, rollout_reward_stats(outcome)


def rollout_reward_stats(outcome: RolloutOutcome) -> dict[str, Any]:
    """Build compact stats for RL callbacks without introducing dataset state."""
    attempt = outcome.attempt
    stats = dict(outcome.stats)
    stats.update(
        {
            "time_cost": float(attempt.timings.get("attempt_seconds", 0.0)),
            "status": attempt.status,
            "task_id": attempt.task_id,
            "attempt_id": attempt.attempt_id,
            "agent_name": attempt.agent_name,
            "episode_id": outcome.episode_id,
            "score": attempt.score,
            "input_tokens": attempt.usage.input_tokens,
            "output_tokens": attempt.usage.output_tokens,
            "total_tokens": attempt.usage.total_tokens,
            "quality_tags": list(attempt.quality_tags),
            "error_count": len(outcome.errors),
        }
    )
    if outcome.errors:
        stats["errors"] = [error.to_dict() for error in outcome.errors]
    return stats
