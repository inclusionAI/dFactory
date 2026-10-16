"""Repair and verify task-attempt artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from anexus.core.errors import AnexusError, ErrorEnvelope, ErrorKind, as_error_envelope
from anexus.outputs.objects import ObjectStore
from anexus.runtime.store import LocalRunStore
from anexus.tasks.artifacts import ArtifactIndex, ArtifactObject, artifact_index_from_dict


@dataclass(frozen=True)
class ArtifactRepairEntry:
    run_id: str
    agent_name: str
    task_id: str
    attempt_id: str
    object_id: str
    uri: str
    before_status: str
    after_status: str
    action: str
    required: bool
    error: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactRepairReport:
    run_id: str
    agent_names: list[str]
    attempt_count: int
    object_count: int
    updated_attempt_count: int
    uploaded_count: int
    verified_count: int
    failed_required_count: int
    failed_optional_count: int
    entries: list[ArtifactRepairEntry]

    @property
    def failed_count(self) -> int:
        return self.failed_required_count + self.failed_optional_count

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "agent_names": list(self.agent_names),
            "attempt_count": self.attempt_count,
            "object_count": self.object_count,
            "updated_attempt_count": self.updated_attempt_count,
            "uploaded_count": self.uploaded_count,
            "verified_count": self.verified_count,
            "failed_required_count": self.failed_required_count,
            "failed_optional_count": self.failed_optional_count,
            "failed_count": self.failed_count,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def repair_run_artifacts(
    store: LocalRunStore,
    *,
    run_id: str,
    object_store: ObjectStore,
) -> ArtifactRepairReport:
    """Repair artifact manifests for all locally stored attempts in a run."""

    attempts = store.list_attempts(run_id)
    entries: list[ArtifactRepairEntry] = []
    updated_attempt_count = 0

    for attempt in attempts:
        raw_artifacts = attempt.record.get("artifacts")
        if not isinstance(raw_artifacts, dict):
            continue
        index = artifact_index_from_dict(raw_artifacts)
        repaired_index, attempt_entries = _repair_attempt_index(
            index,
            object_store=object_store,
            run_id=attempt.run_id,
            agent_name=attempt.agent_name,
            task_id=attempt.task_id,
            attempt_id=attempt.attempt_id,
        )
        entries.extend(attempt_entries)
        updated_record = _record_with_repaired_artifacts(attempt.record, repaired_index)
        if updated_record != attempt.record:
            store.write_attempt_record(updated_record)
            updated_attempt_count += 1

    return ArtifactRepairReport(
        run_id=run_id,
        agent_names=list(dict.fromkeys(attempt.agent_name for attempt in attempts)),
        attempt_count=len(attempts),
        object_count=len(entries),
        updated_attempt_count=updated_attempt_count,
        uploaded_count=sum(1 for entry in entries if entry.action == "uploaded"),
        verified_count=sum(1 for entry in entries if entry.after_status == "verified"),
        failed_required_count=sum(1 for entry in entries if entry.after_status == "failed" and entry.required),
        failed_optional_count=sum(1 for entry in entries if entry.after_status == "failed" and not entry.required),
        entries=entries,
    )


def _repair_attempt_index(
    index: ArtifactIndex,
    *,
    object_store: ObjectStore,
    run_id: str,
    agent_name: str,
    task_id: str,
    attempt_id: str,
) -> tuple[ArtifactIndex, list[ArtifactRepairEntry]]:
    objects: list[ArtifactObject] = []
    entries: list[ArtifactRepairEntry] = []
    for artifact in index.objects:
        repaired, action = _repair_artifact_object(artifact, object_store)
        objects.append(repaired)
        entries.append(
            ArtifactRepairEntry(
                run_id=run_id,
                agent_name=agent_name,
                task_id=task_id,
                attempt_id=attempt_id,
                object_id=artifact.object_id,
                uri=repaired.uri,
                before_status=artifact.status,
                after_status=repaired.status,
                action=action,
                required=repaired.required_for_attempt_record,
                error=repaired.error.to_dict() if repaired.error else None,
            )
        )
    return ArtifactIndex(manifest_uri=index.manifest_uri, objects=objects), entries


def _repair_artifact_object(
    artifact: ArtifactObject,
    object_store: ObjectStore,
) -> tuple[ArtifactObject, str]:
    if artifact.status == "skipped":
        return artifact, "skipped"

    if artifact.status in {"uploaded", "verified"}:
        try:
            stored = object_store.stat(artifact.uri, kind=artifact.storage_kind)
            if artifact.sha256 and stored.sha256 != artifact.sha256:
                raise ValueError("stored object hash does not match attempt record")
            return (
                replace(
                    artifact,
                    sha256=stored.sha256,
                    size_bytes=stored.size_bytes,
                    status="verified",
                    error=None,
                ),
                "verified" if artifact.status != "verified" else "unchanged",
            )
        except Exception as exc:
            if artifact.local_path and Path(artifact.local_path).exists():
                return _upload_and_verify(artifact, object_store)
            return _failed_artifact(artifact, exc), "failed"

    if artifact.local_path and Path(artifact.local_path).exists():
        return _upload_and_verify(artifact, object_store)

    if not artifact.required_for_attempt_record:
        return replace(artifact, status="skipped", error=None), "skipped"

    return (
        replace(
            artifact,
            status="failed",
            error=ErrorEnvelope(
                kind=ErrorKind.ARTIFACT_ERROR,
                message=f"artifact local source is missing: {artifact.local_path}",
                component="artifact.repair",
                details={"local_path": artifact.local_path, "uri": artifact.uri},
            ),
        ),
        "failed",
    )


def _upload_and_verify(
    artifact: ArtifactObject,
    object_store: ObjectStore,
) -> tuple[ArtifactObject, str]:
    assert artifact.local_path is not None
    try:
        uploaded = object_store.upload(
            artifact.local_path,
            artifact.uri,
            kind=artifact.storage_kind,
        )
        verified = object_store.stat(uploaded.uri, kind=artifact.storage_kind)
        return (
            replace(
                artifact,
                uri=verified.uri,
                sha256=verified.sha256,
                size_bytes=verified.size_bytes,
                status="verified",
                error=None,
            ),
            "uploaded",
        )
    except Exception as exc:
        return _failed_artifact(artifact, exc), "failed"


def _failed_artifact(artifact: ArtifactObject, exc: BaseException) -> ArtifactObject:
    return replace(artifact, status="failed", error=_repair_error(exc, artifact=artifact))


def _repair_error(exc: BaseException, *, artifact: ArtifactObject) -> ErrorEnvelope:
    if isinstance(exc, AnexusError):
        return as_error_envelope(exc, component="artifact.repair")
    return ErrorEnvelope(
        kind=ErrorKind.ARTIFACT_ERROR,
        message=str(exc) or exc.__class__.__name__,
        component="artifact.repair",
        retryable=not isinstance(exc, FileNotFoundError),
        details={
            "object_id": artifact.object_id,
            "uri": artifact.uri,
            "local_path": artifact.local_path,
            "exception_type": exc.__class__.__name__,
        },
    )


def _record_with_repaired_artifacts(
    record: dict[str, object],
    repaired_index: ArtifactIndex,
) -> dict[str, object]:
    updated = dict(record)
    updated["artifacts"] = repaired_index.to_dict()
    artifact_errors = [artifact.error.to_dict() for artifact in repaired_index.objects if artifact.error is not None]
    required_errors = [
        artifact.error.to_dict()
        for artifact in repaired_index.objects
        if artifact.error is not None and artifact.required_for_attempt_record
    ]
    prior_errors = [error for error in list(record.get("errors") or []) if not _is_artifact_error(error)]
    updated["errors"] = [*prior_errors, *artifact_errors]

    quality_tags = [str(tag) for tag in list(record.get("quality_tags") or [])]
    if required_errors:
        if "artifact_pending" not in quality_tags:
            quality_tags.append("artifact_pending")
        if str(updated.get("status")) in {"completed", "artifact_pending"}:
            updated["status"] = "artifact_pending"
    else:
        quality_tags = [tag for tag in quality_tags if tag not in {"artifact_pending", "artifact_error"}]
        if str(updated.get("status")) == "artifact_pending":
            updated["status"] = "completed"
    updated["quality_tags"] = quality_tags
    return updated


def _is_artifact_error(error: object) -> bool:
    if not isinstance(error, dict):
        return False
    return error.get("kind") == ErrorKind.ARTIFACT_ERROR.value or error.get("component") in {
        "artifacts",
        "objects",
        "artifact.repair",
    }
