"""Persist run locks, attempt intents, events, and final records."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - local execution targets POSIX.
    fcntl = None  # type: ignore[assignment]

from anexus.core.errors import ErrorEnvelope, ErrorKind, as_error_envelope
from anexus.core.files import (
    JsonlReadError,
    append_jsonl,
    exclusive_file_lock,
    read_jsonl_objects_recoverable,
    write_json_atomic,
)
from anexus.core.run_paths import (
    attempt_intent_path as local_attempt_intent_path,
)
from anexus.core.run_paths import (
    attempt_intents_directory,
    records_directory,
    run_directory,
    safe_path_part,
)
from anexus.core.run_paths import (
    attempt_record_path as local_attempt_record_path,
)
from anexus.events import EventEnvelope

_fallback_active_lock = Lock()


class RunStateError(ValueError):
    """Raised when a run id is used with an invalid new/resume state."""


class RunAlreadyActive(RuntimeError):
    """Raised when another process is executing the same run id."""


EventReadError = JsonlReadError


@dataclass(frozen=True)
class StoredAttempt:
    """A compact view of an attempt record loaded from disk."""

    run_id: str
    task_id: str
    attempt_id: str
    status: str
    payload_sha256: str
    agent_name: str
    sample_index: int
    execution_index: int
    record: dict[str, Any]


@dataclass(frozen=True)
class StoredAttemptIntent:
    """An execution intent written before external resource allocation."""

    path: str
    run_id: str
    task_id: str
    attempt_id: str
    payload_sha256: str
    agent_name: str
    agent_spec_name: str
    sample_index: int
    execution_index: int
    schema_version: int
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "payload_sha256": self.payload_sha256,
            "agent_name": self.agent_name,
            "agent_spec_name": self.agent_spec_name,
            "sample_index": self.sample_index,
            "execution_index": self.execution_index,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class StoreReadError:
    """Diagnostic for a local JSON object that could not be replayed."""

    path: str
    reason: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "reason": self.reason, "message": self.message}


AttemptRecordReadError = StoreReadError
AttemptIntentReadError = StoreReadError


class LocalRunStore:
    """Cross-process-safe local store for run state and attempt outputs."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def run_dir(self, run_id: str) -> Path:
        return run_directory(self.root, run_id)

    def events_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "events.jsonl"

    def records_dir(self, run_id: str) -> Path:
        return records_directory(self.root, run_id)

    def run_lock_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run_lock.json"

    def attempt_intents_dir(self, run_id: str) -> Path:
        return attempt_intents_directory(self.root, run_id)

    def attempt_intent_path(
        self,
        run_id: str,
        *,
        agent_name: str,
        task_id: str,
        sample_index: int,
        attempt_id: str,
    ) -> Path:
        """Return the collision-free path for one physical execution intent."""
        return local_attempt_intent_path(
            self.root,
            run_id,
            agent_name=agent_name,
            task_id=task_id,
            sample_index=sample_index,
            attempt_id=attempt_id,
        )

    def attempt_record_path(
        self,
        run_id: str,
        *,
        agent_name: str,
        task_id: str,
        attempt_id: str,
    ) -> Path:
        """Return the collision-free path for one final attempt record."""
        return local_attempt_record_path(
            self.root,
            run_id,
            agent_name=agent_name,
            task_id=task_id,
            attempt_id=attempt_id,
        )

    def _guard_path(self, run_id: str, kind: str) -> Path:
        return self.root / "runs" / ".locks" / f"{safe_path_part(run_id)}.{kind}.lock"

    @contextmanager
    def active_run(self, run_id: str) -> Iterator[dict[str, object]]:
        """Hold a non-blocking exclusive advisory lock for one invocation."""
        path = self._guard_path(run_id, "active")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, object] = {
            "run_id": run_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": datetime.now(UTC).isoformat(),
        }
        with path.open("a+", encoding="utf-8") as stream:
            fallback_acquired = False
            try:
                if fcntl is None:  # pragma: no cover
                    fallback_acquired = _fallback_active_lock.acquire(blocking=False)
                    if not fallback_acquired:
                        raise RunAlreadyActive(f"run_id {run_id!r} is already active")
                else:
                    try:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        owner = _read_lock_owner(stream)
                        suffix = f" ({owner})" if owner else ""
                        raise RunAlreadyActive(f"run_id {run_id!r} is already active{suffix}") from exc
                stream.seek(0)
                stream.truncate()
                json.dump(metadata, stream, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
                yield metadata
            finally:
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                elif fallback_acquired:  # pragma: no cover
                    _fallback_active_lock.release()

    def claim_run_lock(self, lock: dict[str, Any], *, resume: bool) -> dict[str, Any]:
        """Create a new RunLock or require an existing one for resume."""
        run_id = str(lock.get("run_id", ""))
        if not run_id:
            raise ValueError("run lock requires run_id")

        path = self.run_lock_path(run_id)
        with exclusive_file_lock(self._guard_path(run_id, "run-lock")):
            exists = path.exists()
            if resume:
                if not exists:
                    raise RunStateError(f"cannot resume run_id {run_id!r}: run_lock.json is missing")
            else:
                if self.run_dir(run_id).exists():
                    raise RunStateError(f"run_id {run_id!r} already exists; resume it or use a new run id")
                write_json_atomic(path, lock)

            stored = _read_json_object(path, label="run lock")
            if stored is None:
                raise RuntimeError(f"run lock was not created: {path}")
            return stored

    def read_run_lock(self, run_id: str) -> dict[str, Any] | None:
        return _read_json_object(self.run_lock_path(run_id), label="run lock")

    def record_event(self, event: EventEnvelope) -> None:
        run_id = event.context.get("run_id")
        if isinstance(run_id, str) and run_id:
            append_jsonl(self.events_path(run_id), event.to_dict())

    def write_attempt_intent(self, intent: Mapping[str, Any]) -> Path:
        """Persist one immutable physical execution intent."""
        run_id = str(intent.get("run_id", ""))
        task_id = str(intent.get("task_id", ""))
        agent_name = str(intent.get("agent_name", ""))
        attempt_id = str(intent.get("attempt_id", ""))
        sample_index = _positive_int(intent.get("sample_index"))
        if not run_id or not agent_name or not task_id or not attempt_id or sample_index < 1:
            raise ValueError("attempt intent requires run_id, agent_name, task_id, attempt_id, and sample_index")
        path = self.attempt_intent_path(
            run_id,
            agent_name=agent_name,
            task_id=task_id,
            sample_index=sample_index,
            attempt_id=attempt_id,
        )
        if path.exists():
            stored = _read_json_object(path, label="attempt intent")
            if stored != dict(intent):
                raise RunStateError(f"attempt intent already exists with different content: {path}")
            return path
        write_json_atomic(path, dict(intent))
        return path

    def write_attempt_record(self, record: dict[str, Any]) -> None:
        """Write an attempt record to its Agent- and task-scoped path."""
        run_id = str(record.get("run_id", ""))
        task_id = str(record.get("task_id", ""))
        agent_name = str(record.get("agent_name", ""))
        attempt_id = str(record.get("attempt_id", ""))
        if not run_id or not agent_name or not task_id or not attempt_id:
            raise ValueError("attempt record requires run_id, agent_name, task_id, and attempt_id")
        path = self.attempt_record_path(
            run_id,
            agent_name=agent_name,
            task_id=task_id,
            attempt_id=attempt_id,
        )
        write_json_atomic(path, record)

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        rows, _errors = _read_jsonl_recoverable(self.events_path(run_id))
        return rows

    def list_event_errors(self, run_id: str) -> list[EventReadError]:
        _rows, errors = _read_jsonl_recoverable(self.events_path(run_id))
        return errors

    def list_attempts(self, run_id: str) -> list[StoredAttempt]:
        attempts: list[StoredAttempt] = []
        for path in _json_paths(self.records_dir(run_id)):
            record, _error = _read_object_recoverable(path, label="attempt record")
            if record is not None:
                stored = _stored_attempt(record, fallback_run_id=run_id)
                if stored is not None:
                    attempts.append(stored)
        return attempts

    def list_attempt_record_errors(self, run_id: str) -> list[AttemptRecordReadError]:
        errors: list[AttemptRecordReadError] = []
        for path in _json_paths(self.records_dir(run_id)):
            _record, error = _read_object_recoverable(path, label="attempt record")
            if error is not None:
                errors.append(error)
        return errors

    def list_attempt_intents(self, run_id: str) -> list[StoredAttemptIntent]:
        intents: list[StoredAttemptIntent] = []
        for path in _json_paths(self.attempt_intents_dir(run_id)):
            intent, _error = _read_object_recoverable(path, label="attempt intent")
            if intent is not None:
                intents.append(_stored_attempt_intent(path, intent))
        return intents

    def list_attempt_intent_errors(self, run_id: str) -> list[AttemptIntentReadError]:
        errors: list[AttemptIntentReadError] = []
        for path in _json_paths(self.attempt_intents_dir(run_id)):
            _intent, error = _read_object_recoverable(path, label="attempt intent")
            if error is not None:
                errors.append(error)
        return errors

    def list_orphan_attempts(self, run_id: str) -> list[dict[str, object]]:
        """Return recordless intents enriched with their last observed resources."""
        record_ids = {
            (attempt.agent_name, attempt.task_id, attempt.attempt_id) for attempt in self.list_attempts(run_id)
        }
        events_by_attempt: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for event in self.list_events(run_id):
            context = event.get("context")
            if not isinstance(context, Mapping):
                continue
            agent_name = str(context.get("agent_name", ""))
            task_id = str(context.get("task_id", ""))
            attempt_id = str(context.get("attempt_id", ""))
            if agent_name and task_id and attempt_id:
                events_by_attempt.setdefault((agent_name, task_id, attempt_id), []).append(event)

        candidates: list[dict[str, object]] = []
        for intent in self.list_attempt_intents(run_id):
            identity = (intent.agent_name, intent.task_id, intent.attempt_id)
            if identity in record_ids:
                continue
            events = events_by_attempt.get(identity, [])
            sandbox_ids = _event_context_values(events, "sandbox_id")
            gateway_ids = _event_context_values(events, "gateway_session_id")
            candidates.append(
                {
                    **intent.to_dict(),
                    "event_count": len(events),
                    "last_event": str(events[-1].get("name", "")) if events else "",
                    "sandbox_ids": sandbox_ids,
                    "gateway_session_ids": gateway_ids,
                }
            )
        return candidates


def record_event_safely(
    store: LocalRunStore,
    event: EventEnvelope,
    publish_failure: Callable[[EventEnvelope], None],
) -> None:
    """Record one event without letting journal failures mask runtime work."""
    try:
        store.record_event(event)
    except OSError as exc:
        publish_failure(
            EventEnvelope(
                name="run_store.event_record_failed",
                context=dict(event.context),
                payload=_event_record_error(exc, event=event).to_dict(),
            )
        )


def _event_record_error(exc: Exception, *, event: EventEnvelope) -> ErrorEnvelope:
    base = as_error_envelope(exc, component="runtime.store")
    kind = ErrorKind.INFRASTRUCTURE_ERROR if base.kind == ErrorKind.UNKNOWN else base.kind
    return replace(
        base,
        kind=kind,
        message=f"run store event record failed: {base.message}",
        component="runtime.store",
        retryable=True,
        details={
            **base.details,
            "failed_event_id": event.event_id,
            "failed_event_name": event.name,
            "failed_event_context": dict(event.context),
            "failed_event_payload": dict(event.payload),
            "original_error": base.to_dict(),
        },
    )


def _read_lock_owner(stream: Any) -> str:
    try:
        stream.seek(0)
        value = json.load(stream)
    except (OSError, ValueError, TypeError):
        return ""
    if not isinstance(value, Mapping):
        return ""
    return ", ".join(
        part
        for part in (
            f"pid={value.get('pid')}" if value.get("pid") else "",
            f"host={value.get('hostname')}" if value.get("hostname") else "",
            f"started={value.get('started_at')}" if value.get("started_at") else "",
        )
        if part
    )


def _stored_attempt(
    record: dict[str, Any],
    *,
    fallback_run_id: str,
) -> StoredAttempt | None:
    run_id = str(record.get("run_id") or fallback_run_id)
    task_id = str(record.get("task_id", ""))
    attempt_id = str(record.get("attempt_id", ""))
    if not run_id or not task_id or not attempt_id:
        return None
    return StoredAttempt(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        status=str(record.get("status", "")),
        payload_sha256=str(record.get("payload_sha256", "")),
        agent_name=str(record.get("agent_name", "")),
        sample_index=_positive_int(record.get("sample_index")),
        execution_index=_positive_int(record.get("execution_index")),
        record=record,
    )


def _stored_attempt_intent(path: Path, intent: Mapping[str, Any]) -> StoredAttemptIntent:
    return StoredAttemptIntent(
        path=str(path),
        run_id=str(intent.get("run_id", "")),
        task_id=str(intent.get("task_id", "")),
        attempt_id=str(intent.get("attempt_id", "")),
        payload_sha256=str(intent.get("payload_sha256", "")),
        agent_name=str(intent.get("agent_name", "")),
        agent_spec_name=str(intent.get("agent_spec_name", "")),
        sample_index=_positive_int(intent.get("sample_index")),
        execution_index=_positive_int(intent.get("execution_index")),
        schema_version=int(intent.get("schema_version", 0)),
        created_at=str(intent.get("created_at", "")),
    )


def _read_jsonl_recoverable(path: Path) -> tuple[list[dict[str, Any]], list[EventReadError]]:
    return read_jsonl_objects_recoverable(path)


def _json_paths(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.json")) if directory.exists() else []


def _read_json_object(path: Path, *, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {label} must be a JSON object")
    return value


def _read_object_recoverable(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any] | None, StoreReadError | None]:
    try:
        return _read_json_object(path, label=label), None
    except ValueError as exc:
        reason = "invalid_shape" if "must be a JSON object" in str(exc) else "invalid_json"
        return None, StoreReadError(path=str(path), reason=reason, message=str(exc))


def _positive_int(value: object) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _event_context_values(events: list[dict[str, Any]], field: str) -> list[str]:
    values: list[str] = []
    for event in events:
        context = event.get("context")
        if not isinstance(context, Mapping):
            continue
        value = str(context.get(field, ""))
        if value and value not in values:
            values.append(value)
    return values
