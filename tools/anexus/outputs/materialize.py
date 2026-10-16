"""Materialize task-attempt records for downstream consumers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from anexus.outputs.summary import select_logical_attempt_records
from anexus.runtime.store import LocalRunStore, StoredAttempt

DEFAULT_MATERIALIZED_STATUSES = ("completed", "cleanup_pending", "artifact_pending")


def materialize_attempt_records(
    records: Iterable[StoredAttempt | Mapping[str, Any]],
    *,
    statuses: tuple[str, ...] = DEFAULT_MATERIALIZED_STATUSES,
) -> list[dict[str, Any]]:
    """Return compact JSON-serializable attempt records for downstream use."""

    accepted = set(statuses)
    physical_records = [item.record if isinstance(item, StoredAttempt) else dict(item) for item in records]
    rows: list[dict[str, Any]] = []
    for record in select_logical_attempt_records(physical_records):
        status = str(record.get("status", ""))
        if status not in accepted:
            continue
        result = dict(record.get("result") or {})
        usage = dict(record.get("usage") or {})
        rows.append(
            {
                "run_id": record.get("run_id"),
                "task_id": record.get("task_id"),
                "version": record.get("version"),
                "attempt_id": record.get("attempt_id"),
                "sample_index": record.get("sample_index", 0),
                "execution_index": record.get("execution_index", 0),
                "agent_name": record.get("agent_name"),
                "agent_spec_name": record.get("agent_spec_name"),
                "status": status,
                "score": record.get("score"),
                "passed": result.get("passed"),
                "payload": dict(record.get("payload") or {}),
                "result": result,
                "usage": usage,
                "llm_traces": list(record.get("llm_traces") or []),
                "fixtures": _list_of_mappings(record.get("fixtures")),
                "artifacts": _mapping(record.get("artifacts")),
                "quality_tags": list(record.get("quality_tags") or []),
                "timings": _mapping(record.get("timings")),
                "errors": list(record.get("errors") or []),
                "payload_sha256": record.get("payload_sha256"),
                "lineage": _mapping(record.get("lineage")),
                "sandbox_id": record.get("sandbox_id", ""),
                "sandbox_provider": record.get("sandbox_provider", ""),
                "gateway_session_id": record.get("gateway_session_id", ""),
                "sandbox_resources": _list_of_mappings(record.get("sandbox_resources")),
            }
        )
    return rows


def write_attempt_records_jsonl(
    records: Iterable[StoredAttempt | Mapping[str, Any]],
    path: str | Path,
    *,
    statuses: tuple[str, ...] = DEFAULT_MATERIALIZED_STATUSES,
) -> Path:
    output_path = Path(path)
    rows = materialize_attempt_records(records, statuses=statuses)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    return output_path


def materialize_run_attempts(
    store: LocalRunStore,
    *,
    run_id: str,
    path: str | Path,
    statuses: tuple[str, ...] = DEFAULT_MATERIALIZED_STATUSES,
) -> Path:
    return write_attempt_records_jsonl(
        store.list_attempts(run_id),
        path,
        statuses=statuses,
    )


def _list_of_mappings(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _mapping(raw: object) -> dict[str, Any]:
    return dict(raw) if isinstance(raw, Mapping) else {}
