"""Run replay helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from anexus.runtime.store import EventReadError, LocalRunStore, StoredAttempt


@dataclass(frozen=True)
class RunReplay:
    """Events and attempt records reconstructed from a run store."""

    run_id: str
    agent_names: tuple[str, ...]
    events: list[dict[str, Any]]
    attempts: list[StoredAttempt]
    event_errors: list[EventReadError] = field(default_factory=list)
    attempts_by_agent_task: dict[tuple[str, str], list[StoredAttempt]] = field(default_factory=dict)


def replay_run(store: LocalRunStore, run_id: str) -> RunReplay:
    attempts = store.list_attempts(run_id)
    agent_names = tuple(dict.fromkeys(attempt.agent_name for attempt in attempts))
    attempts_by_agent_task: dict[tuple[str, str], list[StoredAttempt]] = {}
    for attempt in attempts:
        attempts_by_agent_task.setdefault((attempt.agent_name, attempt.task_id), []).append(attempt)
    return RunReplay(
        run_id=run_id,
        agent_names=agent_names,
        events=store.list_events(run_id),
        event_errors=store.list_event_errors(run_id),
        attempts=attempts,
        attempts_by_agent_task=attempts_by_agent_task,
    )
