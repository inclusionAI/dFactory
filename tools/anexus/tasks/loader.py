"""Task source loading helpers."""

from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from anexus.tasks.spec import TaskSpec


class TaskLoadError(ValueError):
    """Invalid task payload with source location context."""

    def __init__(self, source_uri: str, source_line: int, message: str) -> None:
        self.source_uri = source_uri
        self.source_line = source_line
        self.message = message
        super().__init__(f"{source_uri}:{source_line}: {message}")


def load_jsonl_tasks(
    path: str | Path,
    *,
    run_id: str,
    version: int = 1,
    labels: dict[str, str] | None = None,
) -> list[TaskSpec]:
    """Load strict task specs from a JSONL file."""

    source_path = Path(path)
    source_uri = source_path.resolve().as_uri()
    tasks: list[TaskSpec] = []
    with source_path.open(encoding="utf-8") as stream:
        for source_line, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = _parse_payload(line, source_uri=source_uri, source_line=source_line)
            try:
                tasks.append(
                    TaskSpec.from_payload(
                        run_id=run_id,
                        payload=payload,
                        source_uri=source_uri,
                        version=version,
                        source_line=source_line,
                        labels=labels,
                    )
                )
            except ValueError as exc:
                raise TaskLoadError(source_uri, source_line, str(exc)) from exc
    return tasks


def _parse_payload(line: str, *, source_uri: str, source_line: int) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except JSONDecodeError as exc:
        raise TaskLoadError(source_uri, source_line, f"invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise TaskLoadError(source_uri, source_line, "task payload must be a JSON object")
    return payload
