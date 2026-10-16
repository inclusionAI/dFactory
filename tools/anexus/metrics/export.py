"""Local metrics export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anexus.metrics.reducers import RunMetrics


def metrics_to_otlp_json(
    metrics: RunMetrics,
    *,
    service_name: str = "anexus",
    scope_name: str = "anexus",
    timestamp_unix_nano: int = 0,
) -> dict[str, Any]:
    """Convert reduced run metrics into an OTLP Metrics JSON shape."""

    return {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": [
                        _attribute("service.name", service_name),
                        _attribute("service.namespace", "agent-evaluation"),
                        _attribute("anexus.run_id", metrics.run_id),
                    ]
                },
                "scopeMetrics": [
                    {
                        "scope": {"name": scope_name},
                        "metrics": [
                            _sum_metric(
                                "anexus.events.seen",
                                metrics.events_seen,
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _sum_metric(
                                "anexus.attempts.queued",
                                metrics.attempts_queued,
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _sum_metric(
                                "anexus.attempts.admitted",
                                metrics.attempts_admitted,
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _sum_metric(
                                "anexus.attempts.started",
                                metrics.attempts_started,
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _sum_metric(
                                "anexus.attempts.retrying",
                                metrics.attempts_retrying,
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _sum_metric(
                                "anexus.attempts.failed",
                                metrics.attempts_failed,
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _sum_metric(
                                "anexus.cleanup.pending",
                                metrics.cleanup_pending,
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _sum_metric(
                                "anexus.artifacts.pending",
                                metrics.artifact_pending,
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _sum_metric(
                                "anexus.tokens.input",
                                metrics.input_tokens,
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _sum_metric(
                                "anexus.tokens.output",
                                metrics.output_tokens,
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _sum_metric(
                                "anexus.tokens.total",
                                metrics.total_tokens,
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _sum_metric(
                                "anexus.queue.wait_seconds.total",
                                metrics.queue_wait_seconds_total,
                                unit="s",
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _sum_metric(
                                "anexus.queue.global_wait_seconds.total",
                                metrics.global_wait_seconds_total,
                                unit="s",
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _sum_metric(
                                "anexus.queue.agent_wait_seconds.total",
                                metrics.agent_wait_seconds_total,
                                unit="s",
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _status_count_metric(
                                metrics.status_counts,
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                            _error_kind_count_metric(
                                metrics.error_kind_counts,
                                timestamp_unix_nano=timestamp_unix_nano,
                            ),
                        ],
                    }
                ],
            }
        ]
    }


def write_metrics_json(path: str | Path, metrics: RunMetrics) -> Path:
    """Write replayed run metrics as stable local JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics.to_dict(), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def write_otlp_metrics_json(
    path: str | Path,
    metrics: RunMetrics,
    *,
    service_name: str = "anexus",
    scope_name: str = "anexus",
    timestamp_unix_nano: int = 0,
) -> Path:
    """Write reduced run metrics as local OTLP-compatible JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            metrics_to_otlp_json(
                metrics,
                service_name=service_name,
                scope_name=scope_name,
                timestamp_unix_nano=timestamp_unix_nano,
            ),
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_path


def _sum_metric(
    name: str,
    value: int | float,
    *,
    timestamp_unix_nano: int,
    unit: str = "1",
    attributes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "unit": unit,
        "sum": {
            "aggregationTemporality": "AGGREGATION_TEMPORALITY_CUMULATIVE",
            "isMonotonic": True,
            "dataPoints": [
                {
                    "attributes": attributes or [],
                    "timeUnixNano": str(timestamp_unix_nano),
                    **_number_data_point_value(value),
                }
            ],
        },
    }


def _number_data_point_value(value: int | float) -> dict[str, Any]:
    if isinstance(value, float) and not value.is_integer():
        return {"asDouble": value}
    return {"asInt": str(int(value))}


def _status_count_metric(
    status_counts: dict[str, int],
    *,
    timestamp_unix_nano: int,
) -> dict[str, Any]:
    return {
        "name": "anexus.attempts.status",
        "unit": "1",
        "sum": {
            "aggregationTemporality": "AGGREGATION_TEMPORALITY_CUMULATIVE",
            "isMonotonic": True,
            "dataPoints": [
                {
                    "asInt": str(count),
                    "attributes": [_attribute("anexus.status", status)],
                    "timeUnixNano": str(timestamp_unix_nano),
                }
                for status, count in sorted(status_counts.items())
            ],
        },
    }


def _error_kind_count_metric(
    error_kind_counts: dict[str, int],
    *,
    timestamp_unix_nano: int,
) -> dict[str, Any]:
    return {
        "name": "anexus.errors.kind",
        "unit": "1",
        "sum": {
            "aggregationTemporality": "AGGREGATION_TEMPORALITY_CUMULATIVE",
            "isMonotonic": True,
            "dataPoints": [
                {
                    "asInt": str(count),
                    "attributes": [_attribute("anexus.error.kind", error_kind)],
                    "timeUnixNano": str(timestamp_unix_nano),
                }
                for error_kind, count in sorted(error_kind_counts.items())
            ],
        },
    }


def _attribute(key: str, value: Any) -> dict[str, Any]:
    return {"key": key, "value": _any_value(value)}


def _any_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if value is None:
        return {"stringValue": ""}
    return {"stringValue": str(value)}
