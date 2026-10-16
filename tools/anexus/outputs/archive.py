"""Create portable archives for local run outputs."""

from __future__ import annotations

import json
import tarfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from anexus.outputs.summary import summarize_run_attempts
from anexus.runtime.store import LocalRunStore, safe_path_part


@dataclass(frozen=True)
class RunArchiveEntry:
    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RunArchiveErrorEntry:
    path: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RunArchiveReport:
    run_id: str
    archive_path: str
    entry_count: int
    skipped_count: int
    size_bytes: int
    sha256: str
    diagnostics: dict[str, object]
    entries: list[RunArchiveEntry]
    errors: list[RunArchiveErrorEntry]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "archive_path": self.archive_path,
            "entry_count": self.entry_count,
            "skipped_count": self.skipped_count,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "diagnostics": self.diagnostics,
            "entries": [entry.to_dict() for entry in self.entries],
            "errors": [error.to_dict() for error in self.errors],
        }


def archive_run_outputs(
    store: LocalRunStore,
    *,
    run_id: str,
    path: str | Path,
) -> RunArchiveReport:
    """Create a tar.gz archive from one local run directory."""

    run_dir = store.run_dir(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"run output directory does not exist: {run_dir}")

    archive_path = Path(path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_abs = archive_path.resolve()
    root_name = f"runs/{safe_path_part(run_id)}"
    entries: list[RunArchiveEntry] = []
    errors: list[RunArchiveErrorEntry] = []
    diagnostics = _run_diagnostics(store, run_id)

    with tarfile.open(archive_path, "w:gz") as archive:
        for file_path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
            if file_path.resolve() == archive_abs:
                continue
            relative_path = f"{root_name}/{file_path.relative_to(run_dir).as_posix()}"
            try:
                data = _read_archive_file(file_path)
            except OSError as exc:
                errors.append(_archive_file_error(relative_path, exc))
                continue
            _add_bytes(archive, relative_path, data)
            entries.append(
                RunArchiveEntry(
                    path=relative_path,
                    size_bytes=len(data),
                    sha256=sha256(data).hexdigest(),
                )
            )

        generated = {
            f"{root_name}/run_diagnostics.json": json.dumps(
                diagnostics,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ).encode("utf-8"),
            f"{root_name}/summary.json": json.dumps(
                summarize_run_attempts(store, run_id=run_id).to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ).encode("utf-8"),
        }
        for relative_path, data in generated.items():
            _add_bytes(archive, relative_path, data)
            entries.append(
                RunArchiveEntry(
                    path=relative_path,
                    size_bytes=len(data),
                    sha256=sha256(data).hexdigest(),
                )
            )

        manifest = {
            "run_id": run_id,
            "diagnostics": diagnostics,
            "entries": [entry.to_dict() for entry in entries],
            "errors": [error.to_dict() for error in errors],
            "skipped_count": len(errors),
        }
        manifest_data = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        manifest_path = f"{root_name}/archive_manifest.json"
        _add_bytes(archive, manifest_path, manifest_data)
        entries.append(
            RunArchiveEntry(
                path=manifest_path,
                size_bytes=len(manifest_data),
                sha256=sha256(manifest_data).hexdigest(),
            )
        )

    archive_data = archive_path.read_bytes()
    return RunArchiveReport(
        run_id=run_id,
        archive_path=str(archive_path),
        entry_count=len(entries),
        skipped_count=len(errors),
        size_bytes=len(archive_data),
        sha256=sha256(archive_data).hexdigest(),
        diagnostics=diagnostics,
        entries=entries,
        errors=errors,
    )


def _run_diagnostics(store: LocalRunStore, run_id: str) -> dict[str, object]:
    event_errors = [error.to_dict() for error in store.list_event_errors(run_id)]
    attempt_record_errors = [error.to_dict() for error in store.list_attempt_record_errors(run_id)]
    attempt_intents = store.list_attempt_intents(run_id)
    attempt_intent_errors = [error.to_dict() for error in store.list_attempt_intent_errors(run_id)]
    orphan_candidates = store.list_orphan_attempts(run_id)
    return {
        "event_error_count": len(event_errors),
        "event_errors": event_errors,
        "attempt_record_error_count": len(attempt_record_errors),
        "attempt_record_errors": attempt_record_errors,
        "attempt_intent_count": len(attempt_intents),
        "attempt_intent_error_count": len(attempt_intent_errors),
        "attempt_intent_errors": attempt_intent_errors,
        "attempt_lock_error_count": len(attempt_intent_errors),
        "attempt_lock_errors": attempt_intent_errors,
        "orphan_candidate_count": len(orphan_candidates),
        "orphan_candidates": orphan_candidates,
    }


def _read_archive_file(path: Path) -> bytes:
    return path.read_bytes()


def _archive_file_error(path: str, exc: OSError) -> RunArchiveErrorEntry:
    return RunArchiveErrorEntry(
        path=path,
        error_type=exc.__class__.__name__,
        message=str(exc) or exc.__class__.__name__,
    )


def _add_bytes(archive: tarfile.TarFile, path: str, data: bytes) -> None:
    from io import BytesIO

    info = tarfile.TarInfo(path)
    info.size = len(data)
    info.mtime = 0
    archive.addfile(info, BytesIO(data))
