"""Task evaluation contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from anexus.tasks.spec import EvaluationResult, TaskSpec


@dataclass(frozen=True)
class EvaluationInput:
    """Normalized inputs passed from runtime execution to task evaluation."""

    task: TaskSpec
    agent_text: str
    sandbox_stdout: str
    sandbox_stderr: str
    details: dict[str, Any] = field(default_factory=dict)


class TaskEvaluator(Protocol):
    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult: ...


class StructuredOutputEvaluator:
    """Evaluate attempts from verifier JSON, falling back to sandbox success."""

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        parsed = _parse_json_object(evaluation_input.sandbox_stdout)
        if parsed is not None and _looks_like_evaluation(parsed):
            return EvaluationResult(
                passed=bool(parsed["passed"]),
                score=float(parsed["score"]),
                summary=str(parsed.get("summary", "")),
                details={
                    **_details_from(parsed),
                    **evaluation_input.details,
                    "sandbox_stderr": evaluation_input.sandbox_stderr,
                },
            )
        verifier_exit_code = _optional_int(evaluation_input.details.get("verifier_exit_code"))
        if verifier_exit_code is not None and verifier_exit_code != 0:
            return EvaluationResult(
                passed=False,
                score=0.0,
                summary=(
                    evaluation_input.sandbox_stderr
                    or evaluation_input.sandbox_stdout
                    or f"verifier exited with code {verifier_exit_code}"
                ),
                details={
                    **evaluation_input.details,
                    "sandbox_stdout": evaluation_input.sandbox_stdout,
                    "sandbox_stderr": evaluation_input.sandbox_stderr,
                },
            )
        return EvaluationResult(
            passed=True,
            score=1.0,
            summary=evaluation_input.sandbox_stdout or evaluation_input.agent_text,
            details={
                "sandbox_stdout": evaluation_input.sandbox_stdout,
                "sandbox_stderr": evaluation_input.sandbox_stderr,
            },
        )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    if "\n" in stripped:
        candidates.append(stripped.splitlines()[-1].strip())
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _looks_like_evaluation(value: dict[str, Any]) -> bool:
    return "passed" in value and "score" in value


def _details_from(value: dict[str, Any]) -> dict[str, Any]:
    details = value.get("details", {})
    if isinstance(details, dict):
        return details
    return {"details": details}


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
