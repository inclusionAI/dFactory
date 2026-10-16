"""RL rollout data contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from anexus.core.errors import ErrorEnvelope
from anexus.core.execution import ExecutionRequest
from anexus.tasks.spec import TaskAttemptRecord


@dataclass(frozen=True)
class RewardSpec:
    """Reward extraction policy for a rollout."""

    default_reward: float = 0.0
    success_reward: float | None = None
    failure_reward: float | None = None

    def extract(self, attempt: TaskAttemptRecord) -> float:
        """Extract a scalar reward from a completed attempt record."""
        if attempt.score is not None:
            return attempt.score
        if attempt.status == "completed" and self.success_reward is not None:
            return self.success_reward
        if attempt.status != "completed" and self.failure_reward is not None:
            return self.failure_reward
        return self.default_reward


@dataclass(frozen=True)
class RolloutTaskSpec:
    """A task attempt plus RL episode metadata."""

    request: ExecutionRequest
    episode_id: str
    reward_spec: RewardSpec = field(default_factory=RewardSpec)
    rollout_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutOutcome:
    """Combine a rollout attempt with reward and diagnostic fields."""

    reward: float
    attempt: TaskAttemptRecord
    episode_id: str
    stats: dict[str, Any]
    errors: list[ErrorEnvelope]

    def to_record(self) -> RolloutRecord:
        """Convert the outcome to its serializable rollout record."""
        return RolloutRecord.from_outcome(self)


@dataclass(frozen=True)
class RolloutRecord:
    """Serializable single-attempt rollout record for RL and distillation."""

    episode_id: str
    run_id: str
    task_id: str
    attempt_id: str
    agent_name: str
    status: str
    reward: float
    score: float | None
    prompt: str
    payload_sha256: str
    usage: dict[str, int]
    traces: list[dict[str, Any]]
    artifacts: dict[str, Any]
    response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[ErrorEnvelope] = field(default_factory=list)

    @classmethod
    def from_outcome(cls, outcome: RolloutOutcome) -> RolloutRecord:
        """Build a serializable record from a rollout outcome."""
        attempt = outcome.attempt
        return cls(
            episode_id=outcome.episode_id,
            run_id=attempt.run_id,
            task_id=attempt.task_id,
            attempt_id=attempt.attempt_id,
            agent_name=attempt.agent_name,
            status=attempt.status,
            reward=outcome.reward,
            score=attempt.score,
            prompt=_prompt_from_payload(attempt.payload),
            payload_sha256=attempt.payload_sha256,
            usage={
                "input_tokens": attempt.usage.input_tokens,
                "output_tokens": attempt.usage.output_tokens,
                "total_tokens": attempt.usage.total_tokens,
            },
            traces=list(attempt.llm_traces),
            artifacts=dict(attempt.artifacts),
            response=_response_from_artifacts(attempt.artifacts),
            metadata=dict(outcome.stats),
            errors=list(outcome.errors),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable rollout record."""
        return {
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "reward": self.reward,
            "score": self.score,
            "prompt": self.prompt,
            "response": self.response,
            "payload_sha256": self.payload_sha256,
            "usage": dict(self.usage),
            "traces": list(self.traces),
            "artifacts": dict(self.artifacts),
            "metadata": dict(self.metadata),
            "errors": [error.to_dict() for error in self.errors],
        }


def _prompt_from_payload(payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    return json.dumps(payload, sort_keys=True)


def _response_from_artifacts(artifacts: dict[str, Any]) -> str:
    objects = artifacts.get("objects", [])
    if not isinstance(objects, list):
        return ""
    for item in objects:
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "agent.output":
            continue
        local_path = item.get("local_path")
        if not isinstance(local_path, str) or not local_path:
            return ""
        try:
            from pathlib import Path

            path = Path(local_path)
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
    return ""
