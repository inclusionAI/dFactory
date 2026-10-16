"""Object store and binding contracts."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import unquote, urlsplit

from anexus.core.errors import AnexusError, ErrorEnvelope, ErrorKind, as_error_envelope
from anexus.tasks.artifacts import ArtifactIndex, ArtifactObject

ObjectKind = Literal["file", "directory"]


@dataclass(frozen=True)
class ObjectBinding:
    """Catalog binding between an object URI and a sandbox target path."""

    target_path: str
    object_uri: str
    mode: str = "r"
    kind: ObjectKind = "directory"
    optional: bool = False

    @property
    def reads(self) -> bool:
        return "r" in self.mode or "+" in self.mode

    @property
    def writes(self) -> bool:
        return "w" in self.mode or "a" in self.mode or "+" in self.mode

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target_path,
            "object_uri": self.object_uri,
            "oss_uri": self.object_uri,
            "mode": self.mode,
            "kind": self.kind,
            "is_file": self.kind == "file",
            "optional": self.optional,
        }


@dataclass(frozen=True)
class StoredObject:
    uri: str
    local_path: str
    kind: ObjectKind
    sha256: str
    size_bytes: int


class ObjectStore(Protocol):
    def stat(
        self,
        uri: str,
        kind: ObjectKind | None = None,
    ) -> StoredObject: ...

    def fetch(
        self,
        uri: str,
        target_path: str | Path,
        kind: ObjectKind | None = None,
    ) -> StoredObject: ...

    def upload(
        self,
        source_path: str | Path,
        uri: str,
        kind: ObjectKind | None = None,
    ) -> StoredObject: ...


class LocalObjectStore:
    """File-backed object store for local CLI runs."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def stat(
        self,
        uri: str,
        kind: ObjectKind | None = None,
    ) -> StoredObject:
        if urlsplit(uri).scheme == "oss":
            return _stat_oss_object(uri, self.root, kind=kind)
        path = self.path_for_uri(uri)
        if not path.exists():
            raise FileNotFoundError(f"object does not exist: {uri}")
        object_kind = kind or _infer_kind(path)
        _validate_path_kind(path, object_kind)
        return _stored_object(uri=uri, path=path, kind=object_kind)

    def fetch(
        self,
        uri: str,
        target_path: str | Path,
        kind: ObjectKind | None = None,
    ) -> StoredObject:
        if urlsplit(uri).scheme == "oss":
            return _fetch_oss_object(uri, target_path, kind=kind)
        source = self.path_for_uri(uri)
        if not source.exists():
            raise FileNotFoundError(f"object does not exist: {uri}")
        target = Path(target_path)
        object_kind = kind or _infer_kind(source)
        _copy_path(source=source, target=target, kind=object_kind)
        return _stored_object(uri=uri, path=target, kind=object_kind)

    def upload(
        self,
        source_path: str | Path,
        uri: str,
        kind: ObjectKind | None = None,
    ) -> StoredObject:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"upload source does not exist: {source}")
        target = self.path_for_uri(uri)
        object_kind = kind or _infer_kind(source)
        _copy_path(source=source, target=target, kind=object_kind)
        return _stored_object(uri=uri, path=target, kind=object_kind)

    def path_for_uri(self, uri: str) -> Path:
        parts = urlsplit(uri)
        if parts.scheme == "file":
            return Path(unquote(parts.path))
        if parts.scheme == "local":
            key = f"{parts.netloc}{parts.path}".lstrip("/")
            return _safe_store_path(self.root, key)
        if parts.scheme == "":
            return _safe_store_path(self.root, uri)
        raise ValueError(f"unsupported object URI scheme: {parts.scheme}: {uri}")


def _stat_oss_object(
    uri: str,
    root: Path,
    *,
    kind: ObjectKind | None,
) -> StoredObject:
    bucket, bucket_name, key, _oss2 = _oss_bucket(uri)
    object_kind = kind or _infer_oss_kind(key)
    cache_path = _safe_store_path(root, f"_oss_cache/{bucket_name}/{key}")
    _download_oss(bucket, key, cache_path, object_kind, uri=uri, oss2_module=_oss2)
    return _stored_object(uri=uri, path=cache_path, kind=object_kind)


def _fetch_oss_object(
    uri: str,
    target_path: str | Path,
    *,
    kind: ObjectKind | None,
) -> StoredObject:
    bucket, _bucket_name, key, _oss2 = _oss_bucket(uri)
    object_kind = kind or _infer_oss_kind(key)
    target = Path(target_path)
    _download_oss(bucket, key, target, object_kind, uri=uri, oss2_module=_oss2)
    return _stored_object(uri=uri, path=target, kind=object_kind)


def _oss_bucket(uri: str) -> tuple[Any, str, str, Any]:
    parts = urlsplit(uri)
    bucket_name = parts.netloc or os.environ.get("OSS_BUCKET", "")
    key = unquote(parts.path.lstrip("/"))
    if not bucket_name:
        raise ValueError(f"oss:// URI requires a bucket name: {uri}")
    if not key:
        raise ValueError(f"oss:// URI requires an object key: {uri}")

    missing = [name for name in ("OSS_ENDPOINT", "OSS_KEY_ID", "OSS_KEY_SECRET") if not os.environ.get(name)]
    if missing:
        raise RuntimeError("oss:// object requires environment variables: " + ", ".join(missing))

    try:
        import oss2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("oss:// object support requires the Python package 'oss2'") from exc

    endpoint = os.environ["OSS_ENDPOINT"].strip()
    auth = oss2.Auth(os.environ["OSS_KEY_ID"], os.environ["OSS_KEY_SECRET"])
    return oss2.Bucket(auth, endpoint, bucket_name), bucket_name, key, oss2


def _infer_oss_kind(key: str) -> ObjectKind:
    return "directory" if key.endswith("/") else "file"


def _download_oss(
    bucket: Any,
    key: str,
    target: Path,
    kind: ObjectKind,
    *,
    uri: str,
    oss2_module: Any,
) -> None:
    try:
        if kind == "directory":
            _download_oss_directory(bucket, key, target, uri=uri, oss2_module=oss2_module)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        bucket.get_object_to_file(key, str(target))
    except Exception as exc:
        raise RuntimeError(f"failed to fetch OSS object {uri}: {exc}") from exc


def _download_oss_directory(
    bucket: Any,
    key: str,
    target: Path,
    *,
    uri: str,
    oss2_module: Any,
) -> None:
    prefix = key if key.endswith("/") else f"{key}/"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for obj in oss2_module.ObjectIterator(bucket, prefix=prefix):
        object_key = str(obj.key)
        if object_key.endswith("/"):
            continue
        relative = object_key[len(prefix) :]
        if not relative:
            continue
        destination = _safe_store_path(target, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        bucket.get_object_to_file(object_key, str(destination))
        downloaded += 1
    if downloaded == 0:
        raise FileNotFoundError(f"OSS directory did not contain objects: {uri}")


def upload_artifact_index(index: ArtifactIndex, store: ObjectStore) -> ArtifactIndex:
    """Upload collected artifact objects and return an updated index."""

    objects = [_upload_artifact_object(artifact, store) for artifact in index.objects]
    return ArtifactIndex(manifest_uri=index.manifest_uri, objects=objects)


def _upload_artifact_object(artifact: ArtifactObject, store: ObjectStore) -> ArtifactObject:
    if artifact.status != "collected" or artifact.local_path is None:
        return artifact
    try:
        stored = store.upload(artifact.local_path, artifact.uri, kind=artifact.storage_kind)
        return replace(
            artifact,
            uri=stored.uri,
            sha256=stored.sha256,
            size_bytes=stored.size_bytes,
            status="uploaded",
            error=None,
        )
    except Exception as exc:
        return replace(
            artifact,
            status="failed",
            error=_object_error(exc, uri=artifact.uri, local_path=artifact.local_path),
        )


def _object_error(exc: BaseException, *, uri: str, local_path: str) -> ErrorEnvelope:
    if isinstance(exc, AnexusError):
        return as_error_envelope(exc, component="objects")
    return ErrorEnvelope(
        kind=ErrorKind.ARTIFACT_ERROR,
        message=str(exc) or exc.__class__.__name__,
        component="objects",
        retryable=True,
        details={
            "uri": uri,
            "local_path": local_path,
            "exception_type": exc.__class__.__name__,
        },
    )


def _safe_store_path(root: Path, key: str) -> Path:
    candidate = (root / unquote(key)).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"object key escapes store root: {key}")
    return candidate


def _copy_path(*, source: Path, target: Path, kind: ObjectKind) -> None:
    _validate_path_kind(source, kind)
    if kind == "directory":
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _validate_path_kind(path: Path, kind: ObjectKind) -> None:
    if kind == "directory":
        if not path.is_dir():
            raise NotADirectoryError(f"object source is not a directory: {path}")
        return
    if path.is_dir():
        raise IsADirectoryError(f"object source is a directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"object source is not a file: {path}")


def _infer_kind(path: Path) -> ObjectKind:
    return "directory" if path.is_dir() else "file"


def _stored_object(*, uri: str, path: Path, kind: ObjectKind) -> StoredObject:
    _validate_path_kind(path, kind)
    return StoredObject(
        uri=uri,
        local_path=str(path),
        kind=kind,
        sha256=_directory_hash(path) if kind == "directory" else _file_hash(path),
        size_bytes=_directory_size(path) if kind == "directory" else path.stat().st_size,
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
        digest.update(b"path\0")
        digest.update(str(file_path.relative_to(path)).encode())
        digest.update(b"\0sha256\0")
        digest.update(_file_hash(file_path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())
