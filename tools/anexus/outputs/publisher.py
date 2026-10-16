"""Task-attempt output publishers."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from anexus.core.errors import AnexusError, ErrorKind, as_error_envelope
from anexus.core.files import write_json_atomic
from anexus.core.run_paths import attempt_record_path, records_directory
from anexus.tasks.spec import TaskAttemptRecord


class LocalOutputPublisher:
    """Atomically publish one idempotent record per attempt identity."""

    def __init__(
        self,
        root: str | Path,
        *,
        publish_attempts: int = 3,
        initial_retry_seconds: float = 0.05,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize local publication and retry settings."""
        if publish_attempts < 1:
            raise ValueError("publish_attempts must be >= 1")
        if initial_retry_seconds < 0:
            raise ValueError("initial_retry_seconds must be >= 0")
        self.root = Path(root)
        self.publish_attempts = publish_attempts
        self.initial_retry_seconds = initial_retry_seconds
        self._sleep = sleep

    def records_dir(self, run_id: str) -> Path:
        """Return the durable record directory for a run."""
        return records_directory(self.root, run_id)

    def publish(self, record: TaskAttemptRecord) -> Path:
        """Publish a record atomically with bounded filesystem retries."""
        for publish_attempt in range(1, self.publish_attempts + 1):
            try:
                return self._publish_once(record)
            except OSError as exc:
                if publish_attempt >= self.publish_attempts:
                    raise _publish_error(
                        exc,
                        record=record,
                        root=self.root,
                        publish_attempts=publish_attempt,
                    ) from exc
                self._sleep(self.initial_retry_seconds * (2 ** (publish_attempt - 1)))
        raise RuntimeError("publish attempts loop produced no result")

    def _publish_once(self, record: TaskAttemptRecord) -> Path:
        record_path = attempt_record_path(
            self.root,
            record.run_id,
            agent_name=record.agent_name,
            task_id=record.task_id,
            attempt_id=record.attempt_id,
        )
        payload = record.to_dict()
        write_json_atomic(record_path, payload)

        return record_path


def _publish_error(
    exc: BaseException,
    *,
    record: TaskAttemptRecord,
    root: Path,
    publish_attempts: int,
) -> AnexusError:
    base = as_error_envelope(exc, component="outputs.publisher")
    return AnexusError(
        f"task attempt record publish failed: {base.message}",
        kind=ErrorKind.INFRASTRUCTURE_ERROR if base.kind == ErrorKind.UNKNOWN else base.kind,
        component="outputs.publisher",
        retryable=True,
        details={
            **base.details,
            "run_id": record.run_id,
            "task_id": record.task_id,
            "version": record.version,
            "attempt_id": record.attempt_id,
            "agent_name": record.agent_name,
            "payload_sha256": record.payload_sha256,
            "status": record.status,
            "publish_attempts": publish_attempts,
            "root": str(root),
            "phase": "outputs.records.publish",
            "original_error": base.to_dict(),
        },
    )
