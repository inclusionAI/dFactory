"""Artifact manifest contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from anexus.core.errors import ErrorEnvelope, ErrorKind


@dataclass(frozen=True)
class ArtifactSpec:
    object_id: str
    kind: str
    local_path: str
    uri: str
    required: bool = True
    source_path: str = ""
    storage_kind: Literal["file", "directory"] | None = None


@dataclass(frozen=True)
class ArtifactObject:
    object_id: str
    kind: str
    uri: str
    storage_kind: Literal["file", "directory"] = "file"
    local_path: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    status: Literal["expected", "collected", "uploaded", "verified", "failed", "skipped"] = "expected"
    required_for_attempt_record: bool = True
    error: ErrorEnvelope | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["error"] = self.error.to_dict() if self.error else None
        return data


@dataclass(frozen=True)
class ArtifactIndex:
    manifest_uri: str
    objects: list[ArtifactObject] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_uri": self.manifest_uri,
            "objects": [obj.to_dict() for obj in self.objects],
        }


def artifact_index_from_dict(data: Mapping[str, Any]) -> ArtifactIndex:
    """Parse a serialized artifact manifest from an attempt record."""

    raw_objects = data.get("objects", [])
    if not isinstance(raw_objects, list):
        raw_objects = []
    return ArtifactIndex(
        manifest_uri=str(data.get("manifest_uri", "")),
        objects=[artifact_object_from_dict(item) for item in raw_objects if isinstance(item, Mapping)],
    )


def artifact_object_from_dict(data: Mapping[str, Any]) -> ArtifactObject:
    status = str(data.get("status", "expected"))
    if status not in {"expected", "collected", "uploaded", "verified", "failed", "skipped"}:
        status = "expected"
    storage_kind = _storage_kind_from_dict(data)
    return ArtifactObject(
        object_id=str(data.get("object_id", "")),
        kind=str(data.get("kind", "")),
        uri=str(data.get("uri", "")),
        storage_kind=storage_kind,
        local_path=str(data["local_path"]) if data.get("local_path") is not None else None,
        sha256=str(data["sha256"]) if data.get("sha256") is not None else None,
        size_bytes=int(data["size_bytes"]) if data.get("size_bytes") is not None else None,
        status=cast(
            Literal["expected", "collected", "uploaded", "verified", "failed", "skipped"],
            status,
        ),
        required_for_attempt_record=bool(data.get("required_for_attempt_record", True)),
        error=_error_from_dict(data.get("error")),
    )


def collect_artifacts(*, manifest_uri: str, artifacts: list[ArtifactSpec]) -> ArtifactIndex:
    return ArtifactIndex(
        manifest_uri=manifest_uri,
        objects=[_collect_artifact(artifact) for artifact in artifacts],
    )


def _collect_artifact(artifact: ArtifactSpec) -> ArtifactObject:
    path = Path(artifact.local_path)
    if not path.exists():
        storage_kind = artifact.storage_kind or "file"
        if artifact.required:
            return ArtifactObject(
                object_id=artifact.object_id,
                kind=artifact.kind,
                uri=artifact.uri,
                storage_kind=storage_kind,
                local_path=str(path),
                status="failed",
                required_for_attempt_record=True,
                error=ErrorEnvelope(
                    kind=ErrorKind.ARTIFACT_ERROR,
                    message=f"required artifact does not exist: {path}",
                    component="artifacts",
                    details={"local_path": str(path)},
                ),
            )
        return ArtifactObject(
            object_id=artifact.object_id,
            kind=artifact.kind,
            uri=artifact.uri,
            storage_kind=storage_kind,
            local_path=str(path),
            status="skipped",
            required_for_attempt_record=False,
        )

    if path.is_dir():
        storage_kind = "directory"
        sha256 = _directory_hash(path)
        size_bytes = _directory_size(path)
    else:
        storage_kind = "file"
        sha256 = _file_hash(path)
        size_bytes = path.stat().st_size
    if artifact.storage_kind is not None and artifact.storage_kind != storage_kind:
        return ArtifactObject(
            object_id=artifact.object_id,
            kind=artifact.kind,
            uri=artifact.uri,
            storage_kind=artifact.storage_kind,
            local_path=str(path),
            status="failed",
            required_for_attempt_record=artifact.required,
            error=ErrorEnvelope(
                kind=ErrorKind.ARTIFACT_ERROR,
                message=(f"artifact storage kind mismatch: expected {artifact.storage_kind}, found {storage_kind}"),
                component="artifacts",
                details={
                    "local_path": str(path),
                    "expected_storage_kind": artifact.storage_kind,
                    "actual_storage_kind": storage_kind,
                },
            ),
        )

    return ArtifactObject(
        object_id=artifact.object_id,
        kind=artifact.kind,
        uri=artifact.uri,
        storage_kind=storage_kind,
        local_path=str(path),
        sha256=sha256,
        size_bytes=size_bytes,
        status="collected",
        required_for_attempt_record=artifact.required,
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode())
        digest.update(_file_hash(file_path).encode())
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def _storage_kind_from_dict(data: Mapping[str, Any]) -> Literal["file", "directory"]:
    raw = str(data.get("storage_kind") or data.get("object_kind") or data.get("object_type") or "").lower()
    if raw in {"file", "directory"}:
        return cast(Literal["file", "directory"], raw)
    semantic_kind = str(data.get("kind", "")).lower()
    if semantic_kind == "directory" or semantic_kind.endswith(".directory"):
        return "directory"
    return "file"


def _error_from_dict(raw: object) -> ErrorEnvelope | None:
    if not isinstance(raw, Mapping):
        return None
    return ErrorEnvelope.from_dict(raw, default_component="artifacts")
