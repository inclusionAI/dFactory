"""Fetch task-attempt artifacts from an object store."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from anexus.core.errors import AnexusError, ErrorEnvelope, ErrorKind, as_error_envelope
from anexus.outputs.objects import ObjectStore
from anexus.runtime.store import LocalRunStore, safe_path_part
from anexus.tasks.artifacts import ArtifactObject, artifact_index_from_dict


@dataclass(frozen=True)
class ArtifactFetchEntry:
    run_id: str
    agent_name: str
    task_id: str
    attempt_id: str
    object_id: str
    uri: str
    status: str
    target_path: str | None
    required: bool
    error: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactFetchReport:
    run_id: str
    agent_names: list[str]
    target_root: str
    attempt_count: int
    fetched_count: int
    skipped_count: int
    failed_required_count: int
    failed_optional_count: int
    manifest_path: str
    entries: list[ArtifactFetchEntry]

    @property
    def failed_count(self) -> int:
        return self.failed_required_count + self.failed_optional_count

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "agent_names": list(self.agent_names),
            "target_root": self.target_root,
            "attempt_count": self.attempt_count,
            "fetched_count": self.fetched_count,
            "skipped_count": self.skipped_count,
            "failed_required_count": self.failed_required_count,
            "failed_optional_count": self.failed_optional_count,
            "failed_count": self.failed_count,
            "manifest_path": self.manifest_path,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def fetch_run_artifacts(
    store: LocalRunStore,
    *,
    run_id: str,
    object_store: ObjectStore,
    target_root: str | Path,
    overwrite: bool = False,
) -> ArtifactFetchReport:
    """Fetch uploaded or verified artifact objects for a run."""

    target_path = Path(target_root)
    attempts = store.list_attempts(run_id)
    entries: list[ArtifactFetchEntry] = []
    for attempt in attempts:
        raw_artifacts = attempt.record.get("artifacts")
        if not isinstance(raw_artifacts, dict):
            continue
        index = artifact_index_from_dict(raw_artifacts)
        for artifact in index.objects:
            entries.append(
                _fetch_artifact(
                    artifact,
                    object_store=object_store,
                    target_root=target_path,
                    run_id=attempt.run_id,
                    agent_name=attempt.agent_name,
                    task_id=attempt.task_id,
                    attempt_id=attempt.attempt_id,
                    overwrite=overwrite,
                )
            )

    manifest_path = target_path / "download_manifest.json"
    report = ArtifactFetchReport(
        run_id=run_id,
        agent_names=list(dict.fromkeys(attempt.agent_name for attempt in attempts)),
        target_root=str(target_path),
        attempt_count=len(attempts),
        fetched_count=sum(1 for entry in entries if entry.status == "fetched"),
        skipped_count=sum(1 for entry in entries if entry.status == "skipped"),
        failed_required_count=sum(1 for entry in entries if entry.status == "failed" and entry.required),
        failed_optional_count=sum(1 for entry in entries if entry.status == "failed" and not entry.required),
        manifest_path=str(manifest_path),
        entries=entries,
    )
    target_path.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return report


def _fetch_artifact(
    artifact: ArtifactObject,
    *,
    object_store: ObjectStore,
    target_root: Path,
    run_id: str,
    agent_name: str,
    task_id: str,
    attempt_id: str,
    overwrite: bool,
) -> ArtifactFetchEntry:
    if artifact.status not in {"uploaded", "verified"}:
        return ArtifactFetchEntry(
            run_id=run_id,
            agent_name=agent_name,
            task_id=task_id,
            attempt_id=attempt_id,
            object_id=artifact.object_id,
            uri=artifact.uri,
            status="skipped",
            target_path=None,
            required=artifact.required_for_attempt_record,
        )

    target_path = _target_path(
        target_root=target_root,
        agent_name=agent_name,
        task_id=task_id,
        attempt_id=attempt_id,
        object_id=artifact.object_id,
    )
    try:
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"target artifact already exists: {target_path}")
        object_store.fetch(artifact.uri, target_path, kind=artifact.storage_kind)
        return ArtifactFetchEntry(
            run_id=run_id,
            agent_name=agent_name,
            task_id=task_id,
            attempt_id=attempt_id,
            object_id=artifact.object_id,
            uri=artifact.uri,
            status="fetched",
            target_path=str(target_path),
            required=artifact.required_for_attempt_record,
        )
    except Exception as exc:
        return ArtifactFetchEntry(
            run_id=run_id,
            agent_name=agent_name,
            task_id=task_id,
            attempt_id=attempt_id,
            object_id=artifact.object_id,
            uri=artifact.uri,
            status="failed",
            target_path=str(target_path),
            required=artifact.required_for_attempt_record,
            error=_fetch_error(exc, artifact=artifact, target_path=target_path).to_dict(),
        )


def _target_path(
    *,
    target_root: Path,
    agent_name: str,
    task_id: str,
    attempt_id: str,
    object_id: str,
) -> Path:
    return (
        target_root
        / safe_path_part(agent_name)
        / safe_path_part(task_id)
        / safe_path_part(attempt_id)
        / safe_path_part(object_id)
    )


def _fetch_error(
    exc: BaseException,
    *,
    artifact: ArtifactObject,
    target_path: Path,
) -> ErrorEnvelope:
    if isinstance(exc, AnexusError):
        return as_error_envelope(exc, component="artifact.fetch")
    return ErrorEnvelope(
        kind=ErrorKind.ARTIFACT_ERROR,
        message=str(exc) or exc.__class__.__name__,
        component="artifact.fetch",
        retryable=not isinstance(exc, FileExistsError),
        details={
            "object_id": artifact.object_id,
            "uri": artifact.uri,
            "target_path": str(target_path),
            "exception_type": exc.__class__.__name__,
        },
    )
