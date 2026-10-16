"""Small file helpers shared by local runtime stores."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, TextIO

try:
    import fcntl
except ImportError:  # pragma: no cover - the supported local runner targets POSIX.
    fcntl = None  # type: ignore[assignment]

_fallback_file_lock = Lock()


@dataclass(frozen=True)
class JsonlReadError:
    """Diagnostic for one JSONL row that could not be parsed as an object."""

    path: str
    line_number: int
    reason: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line_number": self.line_number,
            "reason": self.reason,
            "message": self.message,
        }


@contextmanager
def exclusive_file_lock(path: str | Path) -> Iterator[None]:
    """Hold an exclusive cross-process advisory lock for a local path."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        if fcntl is None:  # pragma: no cover
            with _fallback_file_lock:
                yield
            return

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def locked_append_stream(path: str | Path) -> Iterator[TextIO]:
    """Open a text file for appending while holding an exclusive file lock."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        if fcntl is None:  # pragma: no cover
            with _fallback_file_lock:
                yield stream
            return

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield stream
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def locked_read_stream(path: str | Path) -> Iterator[TextIO]:
    """Open a text file for reading while holding a shared file lock."""

    target = Path(path)
    with target.open(encoding="utf-8") as stream:
        if fcntl is None:  # pragma: no cover
            with _fallback_file_lock:
                yield stream
            return

        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        try:
            yield stream
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file under a cross-process lock."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    with locked_append_stream(path) as stream:
        stream.write(encoded)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL object rows under the same lock used for appends."""

    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with locked_read_stream(target) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{target}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{target}:{line_number}: JSONL row must be an object")
            rows.append(value)
    return rows


def read_jsonl_objects_recoverable(
    path: str | Path,
) -> tuple[list[dict[str, Any]], list[JsonlReadError]]:
    """Read valid JSONL object rows and return diagnostics for invalid rows."""

    target = Path(path)
    if not target.exists():
        return [], []
    rows: list[dict[str, Any]] = []
    errors: list[JsonlReadError] = []
    with locked_read_stream(target) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    JsonlReadError(
                        path=str(target),
                        line_number=line_number,
                        reason="invalid_json",
                        message=f"{target}:{line_number}: invalid JSON: {exc.msg}",
                    )
                )
                continue
            if not isinstance(value, dict):
                errors.append(
                    JsonlReadError(
                        path=str(target),
                        line_number=line_number,
                        reason="invalid_shape",
                        message=f"{target}:{line_number}: JSONL row must be an object",
                    )
                )
                continue
            rows.append(value)
    return rows, errors


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    """Write one JSON object by replacing the destination atomically."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = stream.name
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_path)
