"""Output publishers and object records."""

from anexus.outputs.objects import (
    LocalObjectStore,
    ObjectBinding,
    ObjectKind,
    ObjectStore,
    StoredObject,
    upload_artifact_index,
)
from anexus.outputs.publisher import LocalOutputPublisher

__all__ = [
    "ArtifactFetchEntry",
    "ArtifactFetchReport",
    "ArtifactRepairEntry",
    "ArtifactRepairReport",
    "LocalOutputPublisher",
    "LocalObjectStore",
    "ObjectBinding",
    "ObjectKind",
    "ObjectStore",
    "RunSummary",
    "RunArchiveEntry",
    "RunArchiveErrorEntry",
    "RunArchiveReport",
    "StoredObject",
    "TaskRunSummary",
    "archive_run_outputs",
    "fetch_run_artifacts",
    "materialize_attempt_records",
    "materialize_run_attempts",
    "repair_run_artifacts",
    "summarize_attempt_records",
    "summarize_run_attempts",
    "upload_artifact_index",
    "write_attempt_records_jsonl",
]


def __getattr__(name: str) -> object:
    if name in {
        "materialize_attempt_records",
        "materialize_run_attempts",
        "write_attempt_records_jsonl",
    }:
        from anexus.outputs import materialize

        return getattr(materialize, name)
    if name in {"ArtifactFetchEntry", "ArtifactFetchReport", "fetch_run_artifacts"}:
        from anexus.outputs import fetch

        return getattr(fetch, name)
    if name in {"ArtifactRepairEntry", "ArtifactRepairReport", "repair_run_artifacts"}:
        from anexus.outputs import repair

        return getattr(repair, name)
    if name in {
        "RunArchiveEntry",
        "RunArchiveErrorEntry",
        "RunArchiveReport",
        "archive_run_outputs",
    }:
        from anexus.outputs import archive

        return getattr(archive, name)
    if name in {
        "RunSummary",
        "TaskRunSummary",
        "summarize_attempt_records",
        "summarize_run_attempts",
    }:
        from anexus.outputs import summary

        return getattr(summary, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
