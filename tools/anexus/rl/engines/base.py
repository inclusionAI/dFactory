"""Engine protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from anexus.tasks.spec import TokenUsage


@runtime_checkable
class InferenceEngine(Protocol):
    """Handle to an RL system's inference backend."""

    def get_completion_url(self, session_id: str) -> str:
        """Return the OpenAI chat completions URL for one rollout session."""
        ...

    def get_session_id(self) -> str:
        """Return the current rollout session id for routing and traces."""
        ...


@runtime_checkable
class EngineConnector(Protocol):
    """Factory that connects to an RL inference backend from runtime env."""

    def connect(self, env: dict[str, str]) -> InferenceEngine:
        """Return an inference engine described by trainer-injected env vars."""
        ...


@dataclass(frozen=True)
class EngineOutput:
    text: str
    model: str
    usage: TokenUsage
    trace_refs: list[dict[str, object]]
