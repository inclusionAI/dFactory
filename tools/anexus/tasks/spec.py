"""Task-level contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Literal, cast

from anexus.core.errors import ErrorEnvelope
from anexus.tasks.artifacts import ArtifactSpec
from anexus.tasks.fixtures import FixturePlan, FixtureSpec


def canonical_payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttemptPolicy:
    max_retries: int = 0
    timeout_seconds: float = 300.0
    repeat_id: str | None = None
    min_retry_wait_seconds: float = 0.0
    max_retry_wait_seconds: float = 60.0
    retry_wait_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("AttemptPolicy.max_retries must be >= 0")
        if self.timeout_seconds <= 0:
            raise ValueError("AttemptPolicy.timeout_seconds must be > 0")
        if self.min_retry_wait_seconds < 0:
            raise ValueError("AttemptPolicy.min_retry_wait_seconds must be >= 0")
        if self.max_retry_wait_seconds < self.min_retry_wait_seconds:
            raise ValueError("AttemptPolicy.max_retry_wait_seconds must be >= min_retry_wait_seconds")
        if self.retry_wait_multiplier < 1:
            raise ValueError("AttemptPolicy.retry_wait_multiplier must be >= 1")


@dataclass(frozen=True)
class ArtifactPolicy:
    mode: Literal["full", "logs_only", "result_only", "skip_workspace"] = "full"


@dataclass(frozen=True)
class TaskSpec:
    run_id: str
    task_id: str
    version: int
    agent_name: str
    payload: dict[str, Any]
    payload_sha256: str
    source_uri: str
    source_line: int | None = None
    task_id_source: Literal["task_id", "instance_id", "id", "name", "line_index"] = "task_id"
    labels: dict[str, str] = field(default_factory=dict)
    attempt_policy: AttemptPolicy = field(default_factory=AttemptPolicy)
    artifact_policy: ArtifactPolicy = field(default_factory=ArtifactPolicy)
    artifacts: list[ArtifactSpec] = field(default_factory=list)
    fixtures: FixturePlan = field(default_factory=FixturePlan)

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "run_id")
        _require_non_empty(self.task_id, "task_id")
        if self.version < 0:
            raise ValueError("TaskSpec.version must be >= 0")
        _require_non_empty(self.agent_name, "agent_name")
        _require_non_empty(self.payload_sha256, "payload_sha256")
        _require_non_empty(self.source_uri, "source_uri")
        if "task_id" in self.payload and str(self.payload["task_id"]).strip() != self.task_id:
            raise ValueError("TaskSpec.task_id must match payload['task_id']")

    @classmethod
    def from_payload(
        cls,
        *,
        run_id: str,
        agent_name: str | None = None,
        payload: dict[str, Any],
        source_uri: str,
        version: int = 1,
        source_line: int | None = None,
        labels: dict[str, str] | None = None,
        attempt_policy: AttemptPolicy | None = None,
        artifact_policy: ArtifactPolicy | None = None,
        artifacts: list[ArtifactSpec] | None = None,
        fixtures: FixturePlan | None = None,
    ) -> TaskSpec:
        payload_copy = dict(payload)
        payload_agent_name = _payload_string(payload_copy.get("agent_name"))
        resolved_agent_name = (agent_name or payload_agent_name or "").strip()
        if not resolved_agent_name:
            raise ValueError("payload['agent_name'] is required")
        if payload_agent_name and payload_agent_name != resolved_agent_name:
            raise ValueError("agent_name must match payload['agent_name']")
        payload_copy["agent_name"] = resolved_agent_name
        task_id, source = _task_identity_from_payload(payload_copy)
        if _payload_string(payload_copy.get("task_id")) != task_id:
            payload_copy["task_id"] = task_id
        return cls(
            run_id=run_id,
            task_id=task_id,
            version=version,
            agent_name=resolved_agent_name,
            payload=payload_copy,
            payload_sha256=canonical_payload_hash(payload_copy),
            source_uri=source_uri,
            source_line=source_line,
            task_id_source=source,
            labels=labels or {},
            attempt_policy=attempt_policy or AttemptPolicy(),
            artifact_policy=artifact_policy or ArtifactPolicy(),
            artifacts=artifacts or _artifact_specs_from_payload(payload_copy),
            fixtures=fixtures if fixtures is not None else _fixture_plan_from_payload(payload_copy),
        )

    def prompt(self) -> str:
        prompt = self.payload.get("prompt")
        if isinstance(prompt, str) and prompt:
            return prompt
        return json.dumps(self.payload, sort_keys=True)


def _artifact_specs_from_payload(payload: dict[str, Any]) -> list[ArtifactSpec]:
    raw_artifacts = payload.get("artifacts", [])
    if raw_artifacts is None:
        return []
    if not isinstance(raw_artifacts, list):
        raise ValueError("payload['artifacts'] must be a list when provided")
    artifacts: list[ArtifactSpec] = []
    for index, raw_artifact in enumerate(raw_artifacts):
        if not isinstance(raw_artifact, dict):
            raise ValueError(f"payload['artifacts'][{index}] must be an object")
        try:
            object_id = str(raw_artifact["object_id"])
            kind = str(raw_artifact["kind"])
            local_path = str(raw_artifact["local_path"])
            uri = str(raw_artifact["uri"])
        except KeyError as exc:
            raise ValueError(f"payload['artifacts'][{index}] missing required field {exc.args[0]!r}") from exc
        artifacts.append(
            ArtifactSpec(
                object_id=object_id,
                kind=kind,
                local_path=local_path,
                uri=uri,
                required=bool(raw_artifact.get("required", True)),
                source_path=str(raw_artifact.get("source_path", "")),
                storage_kind=_artifact_storage_kind(raw_artifact),
            )
        )
    return artifacts


def _artifact_storage_kind(raw: dict[str, object]) -> Literal["file", "directory"] | None:
    value = str(raw.get("storage_kind") or raw.get("object_kind") or raw.get("object_type") or "").strip().lower()
    if value in {"file", "directory"}:
        return cast(Literal["file", "directory"], value)
    if value:
        raise ValueError("artifact storage_kind must be 'file' or 'directory'")
    if "is_file" in raw:
        return "file" if bool(raw.get("is_file")) else "directory"
    return None


def _fixture_plan_from_payload(payload: dict[str, Any]) -> FixturePlan:
    fixtures: list[FixtureSpec] = []
    dt_oss = _payload_string(payload.get("dt_oss"))
    if dt_oss:
        fixtures.append(
            FixtureSpec(
                source_path="",
                source_uri=dt_oss,
                target_path="workspace",
                kind="directory",
                required=True,
            )
        )
    fixtures.extend(_mount_fixtures_from_payload(payload.get("mounts", [])))
    fixtures.extend(_explicit_fixtures_from_payload(payload.get("fixtures", [])))
    return FixturePlan(fixtures)


def _mount_fixtures_from_payload(raw_mounts: object) -> list[FixtureSpec]:
    if raw_mounts is None:
        return []
    if not isinstance(raw_mounts, list):
        raise ValueError("payload['mounts'] must be a list when provided")
    fixtures: list[FixtureSpec] = []
    for index, raw_mount in enumerate(raw_mounts):
        if not isinstance(raw_mount, dict):
            raise ValueError(f"payload['mounts'][{index}] must be an object")
        source_uri = _payload_string(raw_mount.get("oss") or raw_mount.get("oss_uri") or raw_mount.get("uri"))
        if not source_uri:
            raise ValueError(f"payload['mounts'][{index}] missing required source URI")
        target = _safe_payload_path(
            _payload_string(raw_mount.get("target")) or "",
            field=f"payload['mounts'][{index}].target",
            prefix="mounts",
        )
        fixtures.append(
            FixtureSpec(
                source_path="",
                source_uri=source_uri,
                target_path=target,
                kind=_fixture_kind(raw_mount),
                required=not bool(raw_mount.get("optional", False)),
            )
        )
    return fixtures


def _explicit_fixtures_from_payload(raw_fixtures: object) -> list[FixtureSpec]:
    if raw_fixtures is None:
        return []
    if not isinstance(raw_fixtures, list):
        raise ValueError("payload['fixtures'] must be a list when provided")
    fixtures: list[FixtureSpec] = []
    for index, raw_fixture in enumerate(raw_fixtures):
        if not isinstance(raw_fixture, dict):
            raise ValueError(f"payload['fixtures'][{index}] must be an object")
        source_uri = _payload_string(
            raw_fixture.get("source_uri")
            or raw_fixture.get("uri")
            or raw_fixture.get("oss_uri")
            or raw_fixture.get("oss")
        )
        source_path = _payload_string(raw_fixture.get("source_path")) or ""
        if not source_uri and not source_path:
            raise ValueError(f"payload['fixtures'][{index}] requires source_uri or source_path")
        target = _safe_payload_path(
            _payload_string(raw_fixture.get("target_path") or raw_fixture.get("target")) or "",
            field=f"payload['fixtures'][{index}].target_path",
            prefix="fixtures" if "target_path" not in raw_fixture else "",
        )
        fixtures.append(
            FixtureSpec(
                source_path=source_path,
                source_uri=source_uri,
                target_path=target,
                kind=_fixture_kind(raw_fixture),
                required=not bool(raw_fixture.get("optional", False)),
            )
        )
    return fixtures


def _fixture_kind(raw: dict[str, object]) -> str:
    kind = str(raw.get("kind", "")).strip().lower()
    if kind in {"file", "directory"}:
        return kind
    return "file" if bool(raw.get("is_file", False)) else "directory"


def _safe_payload_path(raw_path: str, *, field: str, prefix: str = "") -> str:
    parts = [part for part in raw_path.replace("\\", "/").strip().split("/") if part and part != "."]
    if not parts:
        raise ValueError(f"{field} is required")
    if any(part == ".." for part in parts):
        raise ValueError(f"{field} escapes workspace: {raw_path}")
    path = "/".join(parts)
    return f"{prefix}/{path}" if prefix else path


def _payload_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _task_identity_from_payload(
    payload: dict[str, Any],
) -> tuple[str, Literal["task_id", "instance_id"]]:
    task_id = _payload_string(payload.get("task_id"))
    if task_id:
        return task_id, "task_id"
    instance_id = _payload_string(payload.get("instance_id"))
    if instance_id:
        return instance_id, "instance_id"
    raise ValueError("TaskSpec requires payload['task_id'] or payload['instance_id']")


def _require_non_empty(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} is required")


@dataclass(frozen=True)
class EvaluationResult:
    passed: bool
    score: float
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class TaskLineage:
    source_uri: str
    payload_sha256: str
    agent_spec_name: str
    image_digest: str | None = None
    model_ids: list[str] = field(default_factory=list)
    config_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskAttemptRecord:
    run_id: str
    task_id: str
    version: int
    attempt_id: str
    agent_name: str
    agent_spec_name: str
    payload_sha256: str
    status: Literal[
        "completed",
        "failed",
        "timeout",
        "cancelled",
        "artifact_pending",
        "cleanup_pending",
    ]
    score: float | None
    result: EvaluationResult | None
    llm_traces: list[dict[str, Any]]
    fixtures: list[dict[str, Any]]
    artifacts: dict[str, Any]
    usage: TokenUsage
    timings: dict[str, float]
    errors: list[ErrorEnvelope]
    quality_tags: list[str]
    lineage: TaskLineage
    payload: dict[str, Any] = field(default_factory=dict)
    sandbox_id: str = ""
    sandbox_provider: str = ""
    gateway_session_id: str = ""
    sandbox_resources: list[dict[str, Any]] = field(default_factory=list)
    sample_index: int = 0
    execution_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["errors"] = [error.to_dict() for error in self.errors]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskAttemptRecord:
        """Restore a durable attempt record for run-level aggregation."""
        result_data = data.get("result")
        result = None
        if isinstance(result_data, Mapping):
            result = EvaluationResult(
                passed=bool(result_data.get("passed", False)),
                score=float(result_data.get("score", 0.0)),
                summary=str(result_data.get("summary", "")),
                details=_mapping(result_data.get("details")),
            )
        usage_data = _mapping(data.get("usage"))
        lineage_data = _mapping(data.get("lineage"))
        raw_errors = data.get("errors")
        errors = (
            [ErrorEnvelope.from_dict(item) for item in raw_errors if isinstance(item, Mapping)]
            if isinstance(raw_errors, list)
            else []
        )
        return cls(
            run_id=str(data.get("run_id", "")),
            task_id=str(data.get("task_id", "")),
            version=int(data.get("version", 0)),
            attempt_id=str(data.get("attempt_id", "")),
            agent_name=str(data.get("agent_name", "")),
            agent_spec_name=str(data.get("agent_spec_name", "")),
            payload_sha256=str(data.get("payload_sha256", "")),
            status=cast(Any, str(data.get("status", "failed"))),
            score=float(data["score"]) if data.get("score") is not None else None,
            result=result,
            llm_traces=_mapping_list(data.get("llm_traces")),
            fixtures=_mapping_list(data.get("fixtures")),
            artifacts=_mapping(data.get("artifacts")),
            usage=TokenUsage(
                input_tokens=int(usage_data.get("input_tokens", 0)),
                output_tokens=int(usage_data.get("output_tokens", 0)),
                total_tokens=int(usage_data.get("total_tokens", 0)),
            ),
            timings={str(key): float(value) for key, value in _mapping(data.get("timings")).items()},
            errors=errors,
            quality_tags=[str(item) for item in data.get("quality_tags", [])],
            lineage=TaskLineage(
                source_uri=str(lineage_data.get("source_uri", "")),
                payload_sha256=str(lineage_data.get("payload_sha256", data.get("payload_sha256", ""))),
                agent_spec_name=str(lineage_data.get("agent_spec_name", data.get("agent_spec_name", ""))),
                image_digest=(
                    str(lineage_data["image_digest"]) if lineage_data.get("image_digest") is not None else None
                ),
                model_ids=[str(item) for item in lineage_data.get("model_ids", [])],
                config_refs=[str(item) for item in lineage_data.get("config_refs", [])],
            ),
            payload=_mapping(data.get("payload")),
            sandbox_id=str(data.get("sandbox_id", "")),
            sandbox_provider=str(data.get("sandbox_provider", "")),
            gateway_session_id=str(data.get("gateway_session_id", "")),
            sandbox_resources=_mapping_list(data.get("sandbox_resources")),
            sample_index=int(data.get("sample_index", 0)),
            execution_index=int(data.get("execution_index", 0)),
        )


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]
