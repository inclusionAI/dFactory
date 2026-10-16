"""Rollout record materialization without a dataset lifecycle."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from anexus.core.errors import ErrorEnvelope
from anexus.rl.spec import RewardSpec, RolloutOutcome, RolloutRecord
from anexus.tasks.spec import TaskAttemptRecord

DEFAULT_ROLLOUT_STATUSES = ("completed", "cleanup_pending", "artifact_pending")
RolloutRecordInput = RolloutOutcome | RolloutRecord | TaskAttemptRecord | Mapping[str, Any]
SelectionMode = Literal["all", "latest_per_task", "best_reward_per_task"]


@dataclass(frozen=True)
class RolloutRecordSummary:
    """Aggregate over materialized rollout records."""

    record_count: int
    episode_count: int
    task_count: int
    status_counts: dict[str, int]
    reward_sum: float
    reward_mean: float
    reward_min: float | None
    reward_max: float | None
    total_tokens: int
    response_count: int
    error_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def materialize_rollout_records(
    records: Iterable[RolloutRecordInput],
    *,
    reward_spec: RewardSpec | None = None,
    statuses: tuple[str, ...] = DEFAULT_ROLLOUT_STATUSES,
    select: SelectionMode = "all",
) -> list[RolloutRecord]:
    """Return rollout records from outcomes or task-attempt records.

    The unit remains an attempt record. The optional selection policy only chooses
    which attempts to expose to downstream trainers; it does not create dataset
    state.
    """

    accepted = set(statuses)
    materialized = [
        record
        for record in (rollout_record_from_item(item, reward_spec=reward_spec) for item in records)
        if record.status in accepted
    ]
    return _select_rollout_records(materialized, select=select)


def rollout_record_from_item(
    item: RolloutRecordInput,
    *,
    reward_spec: RewardSpec | None = None,
) -> RolloutRecord:
    if isinstance(item, RolloutRecord):
        return item
    if isinstance(item, RolloutOutcome):
        return item.to_record()
    if isinstance(item, TaskAttemptRecord):
        return rollout_record_from_attempt(item, reward_spec=reward_spec)
    return rollout_record_from_mapping(item, reward_spec=reward_spec)


def rollout_record_from_attempt(
    attempt: TaskAttemptRecord,
    *,
    reward_spec: RewardSpec | None = None,
) -> RolloutRecord:
    """Build one rollout record from a task-attempt record."""

    spec = reward_spec or RewardSpec()
    payload = dict(attempt.payload)
    episode_id = _episode_id_from_payload(payload, task_id=attempt.task_id)
    return RolloutRecord(
        episode_id=episode_id,
        run_id=attempt.run_id,
        task_id=attempt.task_id,
        attempt_id=attempt.attempt_id,
        agent_name=attempt.agent_name,
        status=attempt.status,
        reward=spec.extract(attempt),
        score=attempt.score,
        prompt=_prompt_from_payload(payload),
        payload_sha256=attempt.payload_sha256,
        usage={
            "input_tokens": attempt.usage.input_tokens,
            "output_tokens": attempt.usage.output_tokens,
            "total_tokens": attempt.usage.total_tokens,
        },
        traces=list(attempt.llm_traces),
        artifacts=dict(attempt.artifacts),
        response=_response_from_artifacts(attempt.artifacts),
        metadata={
            "version": attempt.version,
            "agent_spec_name": attempt.agent_spec_name,
            "sandbox_id": attempt.sandbox_id,
            "sandbox_provider": attempt.sandbox_provider,
            "gateway_session_id": attempt.gateway_session_id,
            "sandbox_resources": list(attempt.sandbox_resources),
            "quality_tags": list(attempt.quality_tags),
            "timings": dict(attempt.timings),
            "lineage": {
                "source_uri": attempt.lineage.source_uri,
                "payload_sha256": attempt.lineage.payload_sha256,
                "agent_spec_name": attempt.lineage.agent_spec_name,
                "image_digest": attempt.lineage.image_digest,
                "model_ids": list(attempt.lineage.model_ids),
                "config_refs": list(attempt.lineage.config_refs),
            },
        },
        errors=list(attempt.errors),
    )


def rollout_record_from_mapping(
    record: Mapping[str, Any],
    *,
    reward_spec: RewardSpec | None = None,
) -> RolloutRecord:
    """Build one rollout record from a serialized attempt or rollout record."""

    if "reward" in record and "episode_id" in record:
        return _rollout_record_from_serialized(record)

    payload = _mapping(record.get("payload"))
    task_id = str(record.get("task_id", ""))
    score = _optional_float(record.get("score"))
    status = str(record.get("status", ""))
    reward = _reward_from_mapping(
        score=score,
        status=status,
        reward_spec=reward_spec or RewardSpec(),
    )
    return RolloutRecord(
        episode_id=_episode_id_from_payload(payload, task_id=task_id),
        run_id=str(record.get("run_id", "")),
        task_id=task_id,
        attempt_id=str(record.get("attempt_id", "")),
        agent_name=str(record.get("agent_name", "")),
        status=status,
        reward=reward,
        score=score,
        prompt=_prompt_from_payload(payload),
        payload_sha256=str(record.get("payload_sha256", "")),
        usage=_usage_from_mapping(record.get("usage")),
        traces=_list_of_mappings(record.get("llm_traces") or record.get("traces")),
        artifacts=_mapping(record.get("artifacts")),
        response=_response_from_artifacts(_mapping(record.get("artifacts"))),
        metadata={
            "version": record.get("version"),
            "agent_spec_name": record.get("agent_spec_name"),
            "sandbox_id": str(record.get("sandbox_id", "")),
            "sandbox_provider": str(record.get("sandbox_provider", "")),
            "gateway_session_id": str(record.get("gateway_session_id", "")),
            "sandbox_resources": _list_of_mappings(record.get("sandbox_resources")),
            "quality_tags": list(record.get("quality_tags") or []),
            "timings": _mapping(record.get("timings")),
            "lineage": _mapping(record.get("lineage")),
        },
        errors=_errors_from_records(record.get("errors")),
    )


def write_rollout_records_jsonl(
    records: Iterable[RolloutRecordInput],
    path: str | Path,
    *,
    reward_spec: RewardSpec | None = None,
    statuses: tuple[str, ...] = DEFAULT_ROLLOUT_STATUSES,
    select: SelectionMode = "all",
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = materialize_rollout_records(
        records,
        reward_spec=reward_spec,
        statuses=statuses,
        select=select,
    )
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    return output_path


def read_rollout_records_jsonl(path: str | Path) -> list[RolloutRecord]:
    records: list[RolloutRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(_rollout_record_from_serialized(json.loads(line)))
    return records


def summarize_rollout_records(
    records: Iterable[RolloutRecord | Mapping[str, Any]],
) -> RolloutRecordSummary:
    parsed = [item if isinstance(item, RolloutRecord) else _rollout_record_from_serialized(item) for item in records]
    rewards = [record.reward for record in parsed]
    total_tokens = sum(int(record.usage.get("total_tokens", 0)) for record in parsed)
    return RolloutRecordSummary(
        record_count=len(parsed),
        episode_count=len({(record.agent_name, record.episode_id) for record in parsed}),
        task_count=len({(record.agent_name, record.task_id) for record in parsed}),
        status_counts=dict(Counter(record.status for record in parsed)),
        reward_sum=sum(rewards),
        reward_mean=sum(rewards) / len(rewards) if rewards else 0.0,
        reward_min=min(rewards) if rewards else None,
        reward_max=max(rewards) if rewards else None,
        total_tokens=total_tokens,
        response_count=sum(1 for record in parsed if record.response),
        error_count=sum(len(record.errors) for record in parsed),
    )


def _select_rollout_records(
    records: list[RolloutRecord],
    *,
    select: SelectionMode,
) -> list[RolloutRecord]:
    if select == "all":
        return records
    selected: dict[tuple[str, str], RolloutRecord] = {}
    for record in records:
        key = (record.agent_name, record.task_id)
        current = selected.get(key)
        if current is None:
            selected[key] = record
            continue
        if select == "latest_per_task":
            selected[key] = record
            continue
        if select == "best_reward_per_task" and record.reward > current.reward:
            selected[key] = record
    if select not in {"latest_per_task", "best_reward_per_task"}:
        raise ValueError(f"unknown rollout record selection mode: {select}")
    return list(selected.values())


def _rollout_record_from_serialized(record: Mapping[str, Any]) -> RolloutRecord:
    return RolloutRecord(
        episode_id=str(record.get("episode_id", "")),
        run_id=str(record.get("run_id", "")),
        task_id=str(record.get("task_id", "")),
        attempt_id=str(record.get("attempt_id", "")),
        agent_name=str(record.get("agent_name", "")),
        status=str(record.get("status", "")),
        reward=float(record.get("reward", 0.0) or 0.0),
        score=_optional_float(record.get("score")),
        prompt=str(record.get("prompt", "")),
        payload_sha256=str(record.get("payload_sha256", "")),
        usage=_usage_from_mapping(record.get("usage")),
        traces=_list_of_mappings(record.get("traces")),
        artifacts=_mapping(record.get("artifacts")),
        response=str(record.get("response", "")),
        metadata=_mapping(record.get("metadata")),
        errors=_errors_from_records(record.get("errors")),
    )


def _reward_from_mapping(
    *,
    score: float | None,
    status: str,
    reward_spec: RewardSpec,
) -> float:
    if score is not None:
        return score
    if status == "completed" and reward_spec.success_reward is not None:
        return reward_spec.success_reward
    if status != "completed" and reward_spec.failure_reward is not None:
        return reward_spec.failure_reward
    return reward_spec.default_reward


def _episode_id_from_payload(payload: Mapping[str, Any], *, task_id: str) -> str:
    episode_id = payload.get("episode_id")
    if isinstance(episode_id, str) and episode_id:
        return episode_id
    return task_id


def _prompt_from_payload(payload: Mapping[str, Any]) -> str:
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    return json.dumps(dict(payload), sort_keys=True)


def _response_from_artifacts(artifacts: Mapping[str, Any]) -> str:
    objects = artifacts.get("objects", [])
    if not isinstance(objects, list):
        return ""
    for item in objects:
        if not isinstance(item, Mapping) or item.get("kind") != "agent.output":
            continue
        local_path = item.get("local_path")
        if not isinstance(local_path, str) or not local_path:
            return ""
        path = Path(local_path)
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
    return ""


def _usage_from_mapping(raw: object) -> dict[str, int]:
    data = _mapping(raw)
    return {
        "input_tokens": int(data.get("input_tokens", 0) or 0),
        "output_tokens": int(data.get("output_tokens", 0) or 0),
        "total_tokens": int(data.get("total_tokens", 0) or 0),
    }


def _errors_from_records(raw: object) -> list[ErrorEnvelope]:
    if not isinstance(raw, list):
        return []
    return [_error_from_mapping(item) for item in raw if isinstance(item, Mapping)]


def _error_from_mapping(data: Mapping[str, Any]) -> ErrorEnvelope:
    return ErrorEnvelope.from_dict(data)


def _mapping(raw: object) -> dict[str, Any]:
    return dict(raw) if isinstance(raw, Mapping) else {}


def _list_of_mappings(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _optional_float(raw: object) -> float | None:
    if raw is None:
        return None
    return float(raw)
