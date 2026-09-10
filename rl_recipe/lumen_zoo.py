"""Minimal lumen-zoo boundary for running an agentic task in a sandbox."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutRequest:
    task_id: str
    task_type: str
    question: str
    tools: tuple[str, ...]


@dataclass(frozen=True)
class SandboxResult:
    task_id: str
    final_answer: str
    reward: float


class LumenZoo(ABC):
    """Caller-facing interface; sandbox details stay inside lumen-zoo."""

    @abstractmethod
    def run(self, request: RolloutRequest) -> SandboxResult:
        raise NotImplementedError


class DemoLumenZoo(LumenZoo):
    """Deterministic stand-in; provisioning and tool use are intentionally internal."""

    def __init__(self, fixed_answer: str) -> None:
        self.fixed_answer = fixed_answer

    def run(self, request: RolloutRequest) -> SandboxResult:
        if request.task_type != "search" or "search" not in request.tools:
            raise ValueError("this demo accepts search tasks")
        # A real implementation provisions a sandbox and executes the agent loop here.
        return SandboxResult(
            task_id=request.task_id,
            final_answer=self.fixed_answer,
            reward=1.0,
        )
