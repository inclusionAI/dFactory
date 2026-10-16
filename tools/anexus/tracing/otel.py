"""OpenTelemetry tracing and JSON conversion helpers."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, Status, StatusCode

from anexus.events import EventEnvelope
from anexus.gateway.failures import gateway_failure_affects_task
from anexus.tracing.spans import TraceSpan

_ACTIVE_PROVIDER: TracerProvider | None = None


def init_tracer(
    service_name: str,
    output_dir: str | Path,
    *,
    scope_name: str = "anexus",
    service_namespace: str = "agent-evaluation",
    set_global: bool = True,
    resource_attributes: dict[str, object] | None = None,
) -> trace.Tracer:
    """Initialize an OpenTelemetry tracer that writes finished spans to JSONL."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    attributes = {
        "service.name": service_name,
        "service.namespace": service_namespace,
        **(resource_attributes or {}),
    }
    provider = TracerProvider(resource=Resource.create(attributes))
    provider.add_span_processor(SimpleSpanProcessor(_FileSpanExporter(output_path / "otel.jsonl")))

    global _ACTIVE_PROVIDER
    _ACTIVE_PROVIDER = provider
    if set_global:
        trace.set_tracer_provider(provider)
    return provider.get_tracer(scope_name)


def shutdown_tracer() -> None:
    """Flush and shut down the tracer provider created by :func:`init_tracer`."""

    global _ACTIVE_PROVIDER
    provider = _ACTIVE_PROVIDER
    _ACTIVE_PROVIDER = None
    if provider is not None:
        provider.shutdown()


class _FileSpanExporter(SpanExporter):
    """Append OpenTelemetry readable spans to one compact JSON object per line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._shutdown = False

    def export(self, spans: Iterable[ReadableSpan]) -> SpanExportResult:
        if self._shutdown:
            return SpanExportResult.FAILURE
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path.open("a", encoding="utf-8") as file:
                for span in spans:
                    file.write(_span_to_json_line(span) + "\n")
            return SpanExportResult.SUCCESS
        except OSError:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self._shutdown = True


class OtelTraceSink:
    """Event sink that records run and task-attempt spans with explicit parents."""

    def __init__(
        self,
        tracer: trace.Tracer,
        *,
        run_span_name: str = "anexus.run",
        attempt_span_name: str = "anexus.task.attempt",
    ):
        self.tracer = tracer
        self.run_span_name = run_span_name
        self.attempt_span_name = attempt_span_name
        self._lock = threading.Lock()
        self._run_spans: dict[str, Span] = {}
        self._run_contexts: dict[str, trace.Context] = {}
        self._attempt_spans: dict[str, Span] = {}
        self._attempt_contexts: dict[str, trace.Context] = {}
        self._closed_attempts: set[str] = set()

    def emit(self, event: EventEnvelope) -> None:
        data = event.to_dict()
        context = dict(data.get("context") or {})
        payload = dict(data.get("payload") or {})
        event_name = str(data.get("name") or "")
        timestamp = str(data.get("timestamp") or datetime.now(UTC).isoformat())
        run_id = _string_value(context.get("run_id"))
        attempt_id = _string_value(context.get("attempt_id"))
        attempt_key = _attempt_key(run_id=run_id, context=context, attempt_id=attempt_id)

        with self._lock:
            run_span = self._ensure_run_span(run_id, context)
            if run_span is not None:
                run_span.add_event(event_name, _event_attributes(data), _time_unix_nano(timestamp))

            attempt_span = self._ensure_attempt_span(attempt_key, attempt_id, context, run_id)
            if attempt_span is not None:
                attempt_span.add_event(
                    event_name,
                    _event_attributes(data),
                    _time_unix_nano(timestamp),
                )
                if _event_marks_attempt_error(event_name, context, payload):
                    attempt_span.set_status(Status(StatusCode.ERROR, _error_description(event_name, payload)))
                if _event_closes_attempt(event_name):
                    if not _event_marks_attempt_error(event_name, context, payload):
                        attempt_span.set_status(Status(StatusCode.OK))
                    attempt_span.end(_time_unix_nano(timestamp))
                    self._attempt_spans.pop(attempt_key, None)
                    self._attempt_contexts.pop(attempt_key, None)
                    self._closed_attempts.add(attempt_key)

            if run_span is not None and _event_closes_run(event_name):
                if event_name in {"run.failed", "run.cancelled"}:
                    run_span.set_status(Status(StatusCode.ERROR, _error_description(event_name, payload)))
                else:
                    run_span.set_status(Status(StatusCode.OK))
                run_span.end(_time_unix_nano(timestamp))
                self._run_spans.pop(run_id, None)
                self._run_contexts.pop(run_id, None)

    def close(self) -> None:
        """End any spans that did not receive terminal events."""

        with self._lock:
            attempts = list(self._attempt_spans.values())
            runs = list(self._run_spans.values())
            self._attempt_spans.clear()
            self._attempt_contexts.clear()
            self._run_spans.clear()
            self._run_contexts.clear()
        for span in attempts:
            span.end()
        for span in runs:
            span.end()

    def _ensure_run_span(self, run_id: str, context: dict[str, Any]) -> Span | None:
        if not run_id:
            return None
        span = self._run_spans.get(run_id)
        if span is not None:
            return span
        span = self.tracer.start_span(
            self.run_span_name,
            attributes=_prefixed_attributes(
                {
                    "run_id": run_id,
                    "agent_name": context.get("agent_name", ""),
                    "agent_spec_name": context.get("agent_spec_name", ""),
                }
            ),
        )
        self._run_spans[run_id] = span
        self._run_contexts[run_id] = trace.set_span_in_context(span)
        return span

    def _ensure_attempt_span(
        self,
        attempt_key: str,
        attempt_id: str,
        context: dict[str, Any],
        run_id: str,
    ) -> Span | None:
        if not attempt_key or attempt_key in self._closed_attempts:
            return None
        span = self._attempt_spans.get(attempt_key)
        if span is not None:
            return span
        task_id = _string_value(context.get("task_id"))
        if not run_id or not task_id:
            return None
        parent_context = self._run_contexts.get(run_id)
        span = self.tracer.start_span(
            self.attempt_span_name,
            context=parent_context,
            attributes=_prefixed_attributes(
                {
                    "run_id": run_id,
                    "task_id": task_id,
                    "version": context.get("version", 0),
                    "attempt_id": attempt_id,
                    "agent_name": context.get("agent_name", ""),
                    "agent_spec_name": context.get("agent_spec_name", ""),
                    "gateway_session_id": context.get("gateway_session_id", ""),
                    "sandbox_id": context.get("sandbox_id", ""),
                    "worker_pid": context.get("worker_pid", 0),
                }
            ),
        )
        self._attempt_spans[attempt_key] = span
        self._attempt_contexts[attempt_key] = trace.set_span_in_context(span)
        return span


def spans_to_otlp_json(
    spans: Iterable[TraceSpan],
    *,
    service_name: str = "anexus",
    scope_name: str = "anexus",
) -> dict[str, Any]:
    """Convert spans to the OpenTelemetry JSON shape used by OTLP exporters."""

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        _attribute("service.name", service_name),
                        _attribute("service.namespace", "agent-evaluation"),
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": scope_name},
                        "spans": [_span_to_otlp(span) for span in spans],
                    }
                ],
            }
        ]
    }


def write_otlp_json(
    path: str | Path,
    spans: Iterable[TraceSpan],
    *,
    service_name: str = "anexus",
    scope_name: str = "anexus",
) -> Path:
    """Write reconstructed spans as local OTLP-compatible JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            spans_to_otlp_json(spans, service_name=service_name, scope_name=scope_name),
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_path


def _span_to_otlp(span: TraceSpan) -> dict[str, Any]:
    otel = span.to_otel_dict()
    reserved_attributes = {"run_id", "agent_name", "task_id", "attempt_id"}
    payload: dict[str, Any] = {
        "traceId": _hex_id(span.trace_id, length=32),
        "spanId": _hex_id(span.span_id, length=16),
        "name": span.name,
        "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": str(otel["start_time_unix_nano"]),
        "attributes": [
            _attribute("anexus.run_id", span.run_id),
            _attribute("anexus.agent_name", span.agent_name),
            _attribute("anexus.task_id", span.task_id),
            _attribute("anexus.attempt_id", span.attempt_id),
            *[
                _attribute(f"anexus.{key}", value)
                for key, value in sorted(span.attributes.items())
                if key not in reserved_attributes
            ],
        ],
        "events": [
            {
                "name": str(event.get("name", "")),
                "timeUnixNano": str(_event_time_unix_nano(event)),
                "attributes": [
                    _attribute(key, value) for key, value in sorted(dict(event.get("attributes") or {}).items())
                ],
            }
            for event in span.events
        ],
        "status": {"code": _otel_status_code(span.status)},
    }
    if span.end_time is not None:
        payload["endTimeUnixNano"] = str(otel["end_time_unix_nano"])
    return payload


def _event_time_unix_nano(event: dict[str, Any]) -> int:
    timestamp = event.get("timestamp")
    if timestamp is None:
        return 0
    parsed = datetime.fromisoformat(str(timestamp))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1_000_000_000)


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
    if isinstance(value, dict):
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


def _hex_id(value: str, *, length: int) -> str:
    return sha256(value.encode()).hexdigest()[:length]


def _otel_status_code(status: str) -> str:
    if status == "OK":
        return "STATUS_CODE_OK"
    if status == "ERROR":
        return "STATUS_CODE_ERROR"
    return "STATUS_CODE_UNSET"


def _span_to_json_line(span: ReadableSpan) -> str:
    raw = span.to_json()
    parsed = json.loads(raw)
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _event_attributes(event: dict[str, Any]) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "anexus.event_id": event.get("event_id", ""),
        "anexus.event_name": event.get("name", ""),
    }
    context = event.get("context")
    if isinstance(context, dict):
        attributes.update(_prefixed_attributes(context))
    payload = event.get("payload")
    if isinstance(payload, dict):
        attributes.update(
            {
                f"anexus.payload.{key}": _otel_attribute_value(value)
                for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
                if _string_value(key)
            }
        )
    return attributes


def _prefixed_attributes(values: dict[str, Any]) -> dict[str, Any]:
    return {
        f"anexus.{key}": _otel_attribute_value(value)
        for key, value in sorted(values.items(), key=lambda item: str(item[0]))
        if _string_value(key) and value not in ("", None)
    }


def _otel_attribute_value(value: Any) -> str | bool | int | float | list[str | bool | int | float]:
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple):
        return [_otel_scalar(item) for item in value]
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _otel_scalar(value: Any) -> str | bool | int | float:
    if isinstance(value, str | bool | int | float):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _event_marks_attempt_error(
    event_name: str,
    context: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    status = str(payload.get("status") or "")
    if event_name in {"attempt.failed", "run.failed", "run.cancelled"}:
        return True
    if event_name == "attempt.output_published" and status not in {"", "completed"}:
        return True
    if event_name.endswith(".failed") and event_name != "gateway.trace_export.failed":
        return True
    if event_name.startswith("gateway."):
        return gateway_failure_affects_task(
            task_id=_string_value(context.get("task_id")),
            phase=str(payload.get("phase") or ""),
            failure_context=payload,
        )
    return False


def _event_closes_attempt(event_name: str) -> bool:
    return event_name in {
        "attempt.failed",
        "attempt.output_publish.failed",
        "attempt.output_published",
    }


def _event_closes_run(event_name: str) -> bool:
    return event_name in {"run.completed", "run.failed", "run.cancelled"}


def _error_description(event_name: str, payload: dict[str, Any]) -> str:
    message = str(payload.get("message") or payload.get("error") or "")
    if message:
        return f"{event_name}: {message}"
    reason = str(payload.get("reason") or payload.get("kind") or "")
    if reason:
        return f"{event_name}: {reason}"
    return event_name


def _string_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _attempt_key(*, run_id: str, context: dict[str, Any], attempt_id: str) -> str:
    agent_name = _string_value(context.get("agent_name"))
    task_id = _string_value(context.get("task_id"))
    if not run_id or not task_id or not attempt_id:
        return ""
    return json.dumps(
        [run_id, agent_name, task_id, attempt_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _time_unix_nano(timestamp: str) -> int:
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1_000_000_000)
