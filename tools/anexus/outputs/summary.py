"""Run summaries over task-attempt records."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from math import prod
from typing import Any

from anexus.runtime.store import LocalRunStore


@dataclass(frozen=True)
class TaskRunSummary:
    agent_name: str
    task_id: str
    attempt_count: int
    passed_count: int
    best_score: float | None
    status_counts: dict[str, int]
    pass_at_k: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["pass_at_k"] = {str(key): value for key, value in self.pass_at_k.items()}
        return data


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    task_count: int
    attempt_count: int
    passed_task_count: int
    solved_task_rate: float
    status_counts: dict[str, int]
    pass_at_k: dict[int, float] = field(default_factory=dict)
    tasks: list[TaskRunSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "task_count": self.task_count,
            "attempt_count": self.attempt_count,
            "passed_task_count": self.passed_task_count,
            "solved_task_rate": self.solved_task_rate,
            "status_counts": dict(self.status_counts),
            "pass_at_k": {str(key): value for key, value in self.pass_at_k.items()},
            "tasks": [task.to_dict() for task in self.tasks],
        }


def summarize_run_attempts(
    store: LocalRunStore,
    *,
    run_id: str,
    k_values: Sequence[int] | None = None,
) -> RunSummary:
    """Summarize latest attempt records for one local run."""

    return summarize_attempt_records(
        [attempt.record for attempt in store.list_attempts(run_id)],
        run_id=run_id,
        k_values=k_values,
    )


def summarize_attempt_records(
    records: Iterable[dict[str, Any]],
    *,
    run_id: str = "",
    k_values: Sequence[int] | None = None,
) -> RunSummary:
    """Build task-level run metrics without introducing a dataset lifecycle."""

    records = select_logical_attempt_records(list(records))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    resolved_run_id = run_id
    for record in records:
        task_id = str(record.get("task_id", ""))
        agent_name = str(record.get("agent_name", ""))
        if not agent_name or not task_id:
            continue
        if not resolved_run_id:
            resolved_run_id = str(record.get("run_id", ""))
        grouped[(agent_name, task_id)].append(record)
        status_counts[str(record.get("status", ""))] += 1

    if not grouped:
        return RunSummary(
            run_id=resolved_run_id,
            task_count=0,
            attempt_count=0,
            passed_task_count=0,
            solved_task_rate=0.0,
            status_counts=dict(status_counts),
        )

    min_attempts = min(len(items) for items in grouped.values())
    requested_k_values = _valid_k_values(k_values, max_k=min_attempts)
    tasks = [
        _summarize_task(agent_name, task_id, attempts, k_values=requested_k_values)
        for (agent_name, task_id), attempts in sorted(grouped.items())
    ]
    pass_at_k = {k: sum(task.pass_at_k[k] for task in tasks) / len(tasks) for k in requested_k_values}
    passed_task_count = sum(1 for task in tasks if task.passed_count > 0)
    return RunSummary(
        run_id=resolved_run_id,
        task_count=len(tasks),
        attempt_count=sum(task.attempt_count for task in tasks),
        passed_task_count=passed_task_count,
        solved_task_rate=passed_task_count / len(tasks),
        status_counts=dict(status_counts),
        pass_at_k=pass_at_k,
        tasks=tasks,
    )


def select_logical_attempt_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse v3 physical records to one current record per sample slot."""
    slotted: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        try:
            sample_index = int(record.get("sample_index", 0))
        except (TypeError, ValueError):
            sample_index = 0
        task_id = str(record.get("task_id", ""))
        agent_name = str(record.get("agent_name", ""))
        payload_sha256 = str(record.get("payload_sha256", ""))
        if agent_name and task_id and payload_sha256 and sample_index > 0:
            slotted[(agent_name, task_id, payload_sha256, sample_index)].append(record)
    if not slotted:
        return records

    selected: list[dict[str, Any]] = []
    for slot, candidates in sorted(slotted.items()):
        accepted = [
            candidate
            for candidate in candidates
            if str(candidate.get("status", "")) in {"completed", "cleanup_pending"}
        ]
        if len(accepted) > 1:
            raise ValueError(f"multiple accepted attempt records exist for logical slot {slot!r}")
        selected.append(accepted[0] if accepted else max(candidates, key=_execution_order))
    return selected


def _execution_order(record: dict[str, Any]) -> tuple[int, str]:
    try:
        execution_index = int(record.get("execution_index", 0))
    except (TypeError, ValueError):
        execution_index = 0
    return execution_index, str(record.get("attempt_id", ""))


def _summarize_task(
    agent_name: str,
    task_id: str,
    attempts: list[dict[str, Any]],
    *,
    k_values: Sequence[int],
) -> TaskRunSummary:
    passed_count = sum(1 for attempt in attempts if _attempt_passed(attempt))
    scores = [_score(attempt) for attempt in attempts]
    score_values = [score for score in scores if score is not None]
    return TaskRunSummary(
        agent_name=agent_name,
        task_id=task_id,
        attempt_count=len(attempts),
        passed_count=passed_count,
        best_score=max(score_values) if score_values else None,
        status_counts=dict(Counter(str(attempt.get("status", "")) for attempt in attempts)),
        pass_at_k={k: _pass_at_k_for_task(n=len(attempts), c=passed_count, k=k) for k in k_values},
    )


def _attempt_passed(record: dict[str, Any]) -> bool:
    result = record.get("result")
    if isinstance(result, dict) and isinstance(result.get("passed"), bool):
        return bool(result["passed"])
    score = _score(record)
    return score is not None and score > 0.0


def _score(record: dict[str, Any]) -> float | None:
    value = record.get("score")
    if value is None:
        result = record.get("result")
        if isinstance(result, dict):
            value = result.get("score")
    if value is None:
        return None
    return float(value)


def _valid_k_values(k_values: Sequence[int] | None, *, max_k: int) -> list[int]:
    if max_k < 1:
        return []
    values = k_values if k_values is not None else _default_k_values(max_k)
    return sorted({int(value) for value in values if 1 <= int(value) <= max_k})


def _default_k_values(max_k: int) -> list[int]:
    values = {1}
    power = 2
    while power <= max_k:
        values.add(power)
        power *= 2
    values.update(range(5, max_k + 1, 5))
    return sorted(values)


def _pass_at_k_for_task(*, n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    miss_probability = prod((n - c - index) / (n - index) for index in range(k))
    return 1.0 - miss_probability
