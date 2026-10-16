"""Collision-free local paths for durable run state."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote


def safe_path_part(value: str) -> str:
    """Encode an identifier as one collision-free local path segment."""
    # urllib keeps dots unescaped because they are RFC 3986 unreserved. Encode
    # them explicitly so the special path components "." and ".." can never
    # be produced and so every identifier has a bijective representation.
    return quote(value, safe="").replace(".", "%2E")


def run_directory(root: str | Path, run_id: str) -> Path:
    """Return the durable directory for one run."""
    return Path(root) / "runs" / safe_path_part(run_id)


def records_directory(root: str | Path, run_id: str) -> Path:
    """Return the final attempt record root for one run."""
    return run_directory(root, run_id) / "records"


def attempt_record_path(
    root: str | Path,
    run_id: str,
    *,
    agent_name: str,
    task_id: str,
    attempt_id: str,
) -> Path:
    """Return the Agent- and task-scoped path for one attempt record."""
    return (
        records_directory(root, run_id)
        / safe_path_part(agent_name)
        / safe_path_part(task_id)
        / f"{safe_path_part(attempt_id)}.json"
    )


def attempt_intents_directory(root: str | Path, run_id: str) -> Path:
    """Return the physical execution intent root for one run."""
    return run_directory(root, run_id) / "attempt_locks"


def attempt_intent_path(
    root: str | Path,
    run_id: str,
    *,
    agent_name: str,
    task_id: str,
    sample_index: int,
    attempt_id: str,
) -> Path:
    """Return the Agent-, task-, and sample-scoped path for one intent."""
    return (
        attempt_intents_directory(root, run_id)
        / safe_path_part(agent_name)
        / safe_path_part(task_id)
        / str(sample_index)
        / f"{safe_path_part(attempt_id)}.json"
    )
