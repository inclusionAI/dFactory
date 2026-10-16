"""Read-only run inspection payloads built from stored events and records."""

from __future__ import annotations

from typing import Any

from anexus.outputs import summarize_run_attempts
from anexus.runtime import LocalRunStore
from anexus.ui import build_run_live_state


def build_run_inspection(store: LocalRunStore, *, run_id: str) -> dict[str, object]:
    """Return the same replay-backed run inspection shape used by CLI and API."""

    events = store.list_events(run_id)
    event_errors = store.list_event_errors(run_id)
    attempts = store.list_attempts(run_id)
    attempt_record_errors = store.list_attempt_record_errors(run_id)
    attempt_intents = store.list_attempt_intents(run_id)
    attempt_intent_errors = store.list_attempt_intent_errors(run_id)
    orphan_candidates = store.list_orphan_attempts(run_id)
    run_lock = store.read_run_lock(run_id)
    live_state = build_run_live_state(events)
    summary = summarize_run_attempts(store, run_id=run_id)
    return {
        "run_id": run_id,
        "event_count": len(events),
        "event_error_count": len(event_errors),
        "event_errors": [error.to_dict() for error in event_errors],
        "attempt_record_count": len(attempts),
        "attempt_record_error_count": len(attempt_record_errors),
        "attempt_record_errors": [error.to_dict() for error in attempt_record_errors],
        "attempt_intent_count": len(attempt_intents),
        "attempt_intent_error_count": len(attempt_intent_errors),
        "attempt_intents": [intent.to_dict() for intent in attempt_intents],
        "attempt_intent_errors": [error.to_dict() for error in attempt_intent_errors],
        "attempt_lock_count": len(attempt_intents),
        "attempt_lock_error_count": len(attempt_intent_errors),
        "attempt_locks": [intent.to_dict() for intent in attempt_intents],
        "attempt_lock_errors": [error.to_dict() for error in attempt_intent_errors],
        "orphan_candidate_count": len(orphan_candidates),
        "orphan_candidates": orphan_candidates,
        "run_lock": summarize_run_lock(run_lock),
        "live": live_state.to_dict(),
        "summary": summary.to_dict(),
    }


def build_run_events_payload(store: LocalRunStore, *, run_id: str) -> dict[str, object]:
    """Return stored run events plus recoverable replay diagnostics."""

    events = store.list_events(run_id)
    event_errors = store.list_event_errors(run_id)
    return {
        "run_id": run_id,
        "event_count": len(events),
        "event_error_count": len(event_errors),
        "events": events,
        "event_errors": [error.to_dict() for error in event_errors],
    }


def build_run_attempts_payload(store: LocalRunStore, *, run_id: str) -> dict[str, object]:
    """Return stored attempt records and input-lock diagnostics for one run."""

    attempts = store.list_attempts(run_id)
    attempt_record_errors = store.list_attempt_record_errors(run_id)
    attempt_intents = store.list_attempt_intents(run_id)
    attempt_intent_errors = store.list_attempt_intent_errors(run_id)
    orphan_candidates = store.list_orphan_attempts(run_id)
    return {
        "run_id": run_id,
        "attempt_record_count": len(attempts),
        "attempt_record_error_count": len(attempt_record_errors),
        "attempt_records": [attempt.record for attempt in attempts],
        "attempt_record_errors": [error.to_dict() for error in attempt_record_errors],
        "attempt_intent_count": len(attempt_intents),
        "attempt_intent_error_count": len(attempt_intent_errors),
        "attempt_intents": [intent.to_dict() for intent in attempt_intents],
        "attempt_intent_errors": [error.to_dict() for error in attempt_intent_errors],
        "attempt_lock_count": len(attempt_intents),
        "attempt_lock_error_count": len(attempt_intent_errors),
        "attempt_locks": [intent.to_dict() for intent in attempt_intents],
        "attempt_lock_errors": [error.to_dict() for error in attempt_intent_errors],
        "orphan_candidate_count": len(orphan_candidates),
        "orphan_candidates": orphan_candidates,
    }


def build_run_live_payload(store: LocalRunStore, *, run_id: str) -> dict[str, object]:
    """Return live UI state rebuilt from stored events only."""

    return build_run_live_state(store.list_events(run_id)).to_dict()


def build_run_summary_payload(store: LocalRunStore, *, run_id: str) -> dict[str, object]:
    """Return task-level summary rebuilt from stored attempt records."""

    return summarize_run_attempts(store, run_id=run_id).to_dict()


def summarize_run_lock(run_lock: dict[str, Any] | None) -> dict[str, object]:
    """Return a compact run-lock summary safe for API and CLI output."""

    if run_lock is None:
        return {"exists": False}
    tasks = run_lock.get("tasks")
    agents = run_lock.get("agents")
    return {
        "exists": True,
        "schema_version": run_lock.get("schema_version", 0),
        "fingerprint": str(run_lock.get("fingerprint", "")),
        "agent_names": sorted(str(name) for name in agents) if isinstance(agents, dict) else [],
        "agent_count": len(agents) if isinstance(agents, dict) else 0,
        "n_attempts": int(run_lock.get("n_attempts", 0)),
        "task_count": len(tasks) if isinstance(tasks, list) else 0,
    }
