"""OpenTelemetry Logs JSON conversion for structured log rows."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def logs_to_otlp_json(
    logs: Iterable[Mapping[str, Any]],
    *,
    service_name: str = "anexus",
    scope_name: str = "anexus",
) -> dict[str, Any]:
    """Convert structured Anexus log rows into an OTLP Logs JSON shape."""

    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        _attribute("service.name", service_name),
                        _attribute("service.namespace", "agent-evaluation"),
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": scope_name},
                        "logRecords": [_log_to_otlp(dict(row)) for row in logs],
                    }
                ],
            }
        ]
    }


def write_otlp_logs_json(
    path: str | Path,
    logs: Iterable[Mapping[str, Any]],
    *,
    service_name: str = "anexus",
    scope_name: str = "anexus",
) -> Path:
    """Write structured Anexus log rows as local OTLP-compatible JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            logs_to_otlp_json(logs, service_name=service_name, scope_name=scope_name),
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_path


def _log_to_otlp(row: dict[str, Any]) -> dict[str, Any]:
    level = str(row.get("level", "")).upper()
    payload = {
        "timeUnixNano": str(_time_unix_nano(row.get("timestamp"))),
        "severityText": level,
        "severityNumber": _severity_number(level),
        "body": _any_value(row.get("message", "")),
        "attributes": _log_attributes(row),
    }
    return payload


def _log_attributes(row: dict[str, Any]) -> list[dict[str, Any]]:
    attributes = [
        _attribute("logger.name", row.get("logger", "")),
        _attribute("process.pid", row.get("process_id", 0)),
        _attribute("process.name", row.get("process_name", "")),
        _attribute("thread.name", row.get("thread_name", "")),
    ]
    context = row.get("context")
    if isinstance(context, Mapping):
        attributes.extend(
            _attribute(f"anexus.{key}", value) for key, value in sorted(context.items(), key=lambda item: str(item[0]))
        )
    error = row.get("error")
    if isinstance(error, Mapping):
        attributes.extend(
            _attribute(f"anexus.error.{key}", value)
            for key, value in sorted(error.items(), key=lambda item: str(item[0]))
        )
    fields = row.get("fields")
    if isinstance(fields, Mapping):
        attributes.extend(
            _attribute(f"anexus.field.{key}", value)
            for key, value in sorted(fields.items(), key=lambda item: str(item[0]))
        )
    if row.get("exception"):
        attributes.append(_attribute("exception.stacktrace", row["exception"]))
    return attributes


def _time_unix_nano(timestamp: object) -> int:
    if timestamp is None:
        return 0
    parsed = datetime.fromisoformat(str(timestamp))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1_000_000_000)


def _severity_number(level: str) -> int:
    return {
        "TRACE": 1,
        "DEBUG": 5,
        "INFO": 9,
        "WARNING": 13,
        "WARN": 13,
        "ERROR": 17,
        "CRITICAL": 21,
        "FATAL": 21,
    }.get(level, 0)


def _attribute(key: str, value: Any) -> dict[str, Any]:
    return {"key": key, "value": _any_value(value)}


def _any_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, list | tuple):
        return {"arrayValue": {"values": [_any_value(item) for item in value]}}
    if isinstance(value, Mapping):
        return {
            "kvlistValue": {
                "values": [
                    _attribute(str(key), item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                ]
            }
        }
    if value is None:
        return {"stringValue": ""}
    return {"stringValue": str(value)}
