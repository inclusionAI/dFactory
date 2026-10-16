"""Base interfaces and result types for host-side graders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from anexus.tasks.spec import EvaluationResult


class SandboxCommandResult(Protocol):
    stdout: str
    stderr: str
    exit_code: int


class ArtifactSandbox(Protocol):
    async def read_file(self, path: str) -> str: ...

    async def run_command(self, command: str, *, timeout: int = 60) -> SandboxCommandResult: ...


@dataclass(frozen=True)
class GraderConfig:
    """Host-side grader configuration parsed from a task verifier block."""

    enabled: bool = False
    grader: str = "swebench"
    eval_timeout_seconds: int = 300
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraderResult:
    """Benchmark-agnostic host-side grader result."""

    score: float = 0.0
    passed: bool = False
    error: str = ""
    eval_log: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_evaluation_result(self) -> EvaluationResult:
        return EvaluationResult(
            passed=self.passed,
            score=self.score,
            summary=self.error or self.eval_log,
            details={
                **self.details,
                "eval_log": self.eval_log,
                "grader_error": self.error,
            },
        )


class BaseGrader(ABC):
    """Common interface implemented by host-side verification graders."""

    name: str

    @abstractmethod
    async def extract_artifact(
        self,
        sandbox: ArtifactSandbox,
        payload: dict[str, Any],
        config: GraderConfig | None = None,
    ) -> str:
        """Extract the candidate artifact from the task sandbox."""

    @abstractmethod
    def generate_eval_script(
        self,
        payload: dict[str, Any],
        config: GraderConfig | None = None,
    ) -> str:
        """Generate verifier-side evaluation script text."""

    @abstractmethod
    def build_command(
        self,
        payload: dict[str, Any],
        config: GraderConfig | None = None,
    ) -> str:
        """Build the verifier-side shell command."""

    @abstractmethod
    def grade_output(
        self,
        eval_output: str,
        eval_exit_code: int,
        payload: dict[str, Any],
        config: GraderConfig | None = None,
    ) -> GraderResult:
        """Convert verifier output into a benchmark-agnostic result."""
