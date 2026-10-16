"""Immutable run inputs used to validate safe resume."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from anexus.config import ResolvedAgentPlan, RunPlan
from anexus.config.agents import SandboxSpec, SidecarSpec

RUN_LOCK_SCHEMA_VERSION = 5


class RunLockMismatch(ValueError):
    """Raised when an existing run directory cannot be resumed safely."""


@dataclass(frozen=True)
class RunLock:
    """Stable semantic fingerprint for one run id."""

    schema_version: int
    run_id: str
    catalog_version: str
    loaded_task_source: str
    n_attempts: int
    cluster: dict[str, str] = field(default_factory=dict)
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    trace_storage: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        """Return the persisted lock payload."""
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "catalog_version": self.catalog_version,
            "loaded_task_source": self.loaded_task_source,
            "n_attempts": self.n_attempts,
            "cluster": dict(self.cluster),
            "agents": {name: dict(agent) for name, agent in self.agents.items()},
            "tasks": [dict(task) for task in self.tasks],
            "trace_storage": dict(self.trace_storage),
        }
        if include_fingerprint:
            data["fingerprint"] = self.fingerprint or fingerprint_run_lock(data)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunLock:
        """Restore and validate a persisted lock payload."""
        content = {
            "schema_version": int(data.get("schema_version", 0)),
            "run_id": str(data.get("run_id", "")),
            "catalog_version": str(data.get("catalog_version", "")),
            "loaded_task_source": str(data.get("loaded_task_source", "")),
            "n_attempts": int(data.get("n_attempts", 0)),
            "cluster": _str_dict_value(data.get("cluster")),
            "agents": _named_dict_values(data.get("agents")),
            "tasks": _dict_list(data.get("tasks")),
            "trace_storage": _dict_value(data.get("trace_storage")),
        }
        computed = fingerprint_run_lock(content)
        persisted = str(data.get("fingerprint", ""))
        if content["schema_version"] == RUN_LOCK_SCHEMA_VERSION and persisted and persisted != computed:
            raise RunLockMismatch("stored run lock fingerprint does not match its content")
        return cls(**content, fingerprint=persisted or computed)


def build_run_lock(
    plan: RunPlan,
    *,
    trace_storage: Mapping[str, Any] | None = None,
) -> RunLock:
    """Build a credential-free semantic lock for a resolved plan."""
    content: dict[str, Any] = {
        "schema_version": RUN_LOCK_SCHEMA_VERSION,
        "run_id": plan.run_id,
        "catalog_version": plan.catalog_version or _catalog_version(plan),
        "cluster": {"name": plan.cluster.name, "region": plan.cluster.region},
        "loaded_task_source": plan.loaded_task_source,
        "n_attempts": plan.n_attempts,
        "agents": {agent_name: _agent_lock(agent) for agent_name, agent in plan.agents.items()},
        "tasks": [
            {
                "index": index,
                "task_id": task.task_id,
                "version": task.version,
                "agent_name": task.agent_name,
                "source_uri": task.source_uri,
                "source_line": task.source_line,
                "payload_sha256": task.payload_sha256,
                "max_retries": task.attempt_policy.max_retries,
                "repeat_id": task.attempt_policy.repeat_id or "",
                "timeout_seconds": task.attempt_policy.timeout_seconds,
                "min_retry_wait_seconds": task.attempt_policy.min_retry_wait_seconds,
                "max_retry_wait_seconds": task.attempt_policy.max_retry_wait_seconds,
                "retry_wait_multiplier": task.attempt_policy.retry_wait_multiplier,
                "artifact_policy": task.artifact_policy.mode,
                "fixtures_sha256": _hash_value([asdict(fixture) for fixture in task.fixtures.fixtures]),
                "artifacts_sha256": _hash_value([asdict(artifact) for artifact in task.artifacts]),
            }
            for index, task in enumerate(plan.tasks)
        ],
        "trace_storage": dict(trace_storage or {}),
    }
    return RunLock(**content, fingerprint=fingerprint_run_lock(content))


def _agent_lock(agent: ResolvedAgentPlan) -> dict[str, Any]:
    """Return the credential-free semantic lock for one Agent plan."""
    agent_model_overrides = {
        role: model for role, model in agent.agent_models.items() if role != "main" or model != agent.model_id
    }
    gateway_configuration = asdict(agent.gateway)
    gateway_configuration["environment"] = _semantic_env(agent.gateway.environment)
    gateway_configuration["admin_retry"]["transient_status_codes"] = sorted(
        agent.gateway.admin_retry.transient_status_codes
    )
    for mirror in gateway_configuration["mirrors"]:
        mirror.pop("api_key", None)
    runtime_configuration = asdict(agent.agent_runtime)
    runtime_configuration["environment"] = _semantic_env(agent.agent_runtime.environment)
    return {
        "agent_spec_name": agent.agent_spec.name,
        "model_id": agent.model_id,
        "judge_model_id": agent.judge_model_id,
        "output_prefix": agent.output_prefix,
        "gateway": {
            "name": agent.gateway.name,
            "driver": agent.gateway.driver,
            "sandbox_provider": agent.gateway.sandbox_provider,
            "configuration_sha256": _hash_value(gateway_configuration),
            "resources": {
                "cpu_request": agent.gateway.resources.cpu_request,
                "cpu_limit": agent.gateway.resources.cpu_limit,
                "memory_request": agent.gateway.resources.memory_request,
                "memory_limit": agent.gateway.resources.memory_limit,
            },
            "mirrors": [
                {
                    "name": mirror.name,
                    "models": list(mirror.models),
                    "protocol": mirror.protocol or "",
                    "upstream_url": _normalize_url(mirror.upstream_url),
                }
                for mirror in agent.gateway.mirrors
            ],
            **({"agent_models": agent_model_overrides} if agent_model_overrides else {}),
        },
        "agent_runtime": {
            "name": agent.agent_runtime.name,
            "driver": agent.agent_runtime.driver,
            "protocol": agent.agent_runtime.protocol,
            "configuration_sha256": _hash_value(runtime_configuration),
        },
        "sandbox": _sandbox_lock(agent.sandbox, agent_env=agent.agent_spec.env),
        "sidecars": [_sidecar_lock(sidecar) for sidecar in agent.sidecars],
    }


def trace_storage_identity(
    *,
    endpoint: str,
    bucket: str,
    prefix: str,
) -> dict[str, str]:
    """Return the non-secret trace destination identity locked across resume."""
    return {
        "endpoint": _normalize_url(endpoint),
        "bucket": bucket.strip(),
        "prefix": prefix.strip("/"),
    }


def fingerprint_run_lock(data: Mapping[str, Any]) -> str:
    """Return a stable hash for run-lock content."""
    payload = {key: value for key, value in data.items() if key != "fingerprint"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def require_matching_run_lock(existing: RunLock, current: RunLock) -> None:
    """Raise when a stored run lock is legacy or differs from the current plan."""
    if existing.schema_version != RUN_LOCK_SCHEMA_VERSION:
        raise RunLockMismatch(
            f"run_id {current.run_id!r} uses legacy RunLock schema "
            f"{existing.schema_version}; schema {RUN_LOCK_SCHEMA_VERSION} is required for resume"
        )
    if existing.fingerprint == current.fingerprint:
        return

    existing_data = existing.to_dict(include_fingerprint=False)
    current_data = current.to_dict(include_fingerprint=False)
    changed = [
        key for key in sorted(set(existing_data) | set(current_data)) if existing_data.get(key) != current_data.get(key)
    ]
    changed_text = ", ".join(changed[:8]) if changed else "fingerprint"
    raise RunLockMismatch(
        f"run lock mismatch for run_id {current.run_id!r}: "
        f"existing={existing.fingerprint[:12]} current={current.fingerprint[:12]}; "
        f"differing fields: {changed_text}. Use a new run_id or the original plan."
    )


def _sandbox_lock(spec: SandboxSpec, *, agent_env: Mapping[str, str]) -> dict[str, Any]:
    return {
        "provider": spec.provider,
        "image": spec.image,
        "digest": spec.digest,
        "port": spec.port,
        "cpu_request": spec.cpu_request,
        "cpu_limit": spec.cpu_limit,
        "memory_request_mb": spec.memory_request_mb,
        "memory_limit_mb": spec.memory_limit_mb,
        "command": spec.command or "",
        "task_runner": spec.task_runner,
        "image_from_payload": spec.image_from_payload,
        "env": _semantic_env({**agent_env, **spec.env}),
        "file_overrides_sha256": _hash_value(spec.file_overrides),
        "verify_sha256": _hash_value(spec.verify or {}),
        "network_policy": spec.network_policy.to_dict(),
        "bindings_sha256": _hash_value(
            [
                {
                    "object_uri": binding.object_uri,
                    "target_path": binding.target_path,
                    "kind": binding.kind,
                    "optional": binding.optional,
                }
                for binding in spec.bindings
            ]
        ),
        "extra_config_sha256": _hash_value(spec.extra_config or {}),
    }


def _sidecar_lock(spec: SidecarSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "provider": spec.provider,
        "image": spec.image,
        "digest": spec.digest,
        "port": spec.port,
        "ports": list(spec.ports),
        "command": spec.command or "",
        "env_url_key": spec.env_url_key,
        "task_filter": spec.task_filter,
        "health_path": spec.health_path,
        "health_timeout_seconds": spec.health_timeout_seconds,
        "inject_kubeconfig": spec.inject_kubeconfig,
        "kubeconfig": spec.kubeconfig,
        "cpu_request": spec.cpu_request,
        "cpu_limit": spec.cpu_limit,
        "memory_request_mb": spec.memory_request_mb,
        "memory_limit_mb": spec.memory_limit_mb,
        "needs_provider": spec.needs_provider,
        "network_policy": spec.network_policy.to_dict(),
        "env_from_payload": dict(spec.env_from_payload),
        "env": _semantic_env(spec.env),
        "extra_config_sha256": _hash_value(spec.extra_config or {}),
    }


def _semantic_env(env: Mapping[str, object]) -> dict[str, str]:
    secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
    return {
        str(key): str(value)
        for key, value in sorted(env.items())
        if not any(marker in str(key).upper() for marker in secret_markers)
    }


def _catalog_version(plan: RunPlan) -> str:
    versions = {task.labels.get("catalog_version", "") for task in plan.tasks}
    versions.discard("")
    return next(iter(versions)) if len(versions) == 1 else ""


def _normalize_url(value: str) -> str:
    raw = value.strip().rstrip("/")
    if not raw:
        return ""
    parts = urlsplit(raw if "://" in raw else f"//{raw}")
    normalized = urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path,
            parts.query,
            parts.fragment,
        )
    )
    return normalized if parts.scheme else normalized.removeprefix("//")


def _hash_value(value: Any) -> str:
    if value in ({}, [], None, ""):
        return ""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dict_value(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _str_dict_value(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _named_dict_values(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): dict(item) for key, item in value.items() if isinstance(item, Mapping)}


def _dict_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]
