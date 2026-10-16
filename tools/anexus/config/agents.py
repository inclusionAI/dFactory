"""Agent specification contracts and parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from anexus.outputs.objects import ObjectBinding
from anexus.sandbox.network import SandboxNetworkPolicy, sandbox_network_policy_from_config

DEFAULT_PORT = 8080
DEFAULT_CPU_REQUEST = 250
DEFAULT_CPU_LIMIT = 4000
DEFAULT_MEMORY_REQUEST_MB = 1024
DEFAULT_MEMORY_LIMIT_MB = 8192
DEFAULT_TASK_TIMEOUT_SECONDS = 1500.0
DEFAULT_MIN_RETRY_WAIT_SECONDS = 0.0
DEFAULT_MAX_RETRY_WAIT_SECONDS = 60.0
DEFAULT_RETRY_WAIT_MULTIPLIER = 1.0


def _request_or_default(raw: dict[str, Any], key: str, default: int, limit: int) -> int:
    if key in raw:
        return int(raw[key])
    return min(default, limit) if limit > 0 else default


@dataclass(frozen=True)
class SandboxSpec:
    """Sandbox container and resource request for an agent."""

    image: str
    provider: str = "akernel"
    port: int = DEFAULT_PORT
    cpu_request: int = DEFAULT_CPU_REQUEST
    cpu_limit: int = DEFAULT_CPU_LIMIT
    memory_request_mb: int = DEFAULT_MEMORY_REQUEST_MB
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB
    command: str | None = None
    task_runner: str | list[str] = ""
    image_from_payload: str = ""
    digest: str = ""
    env: dict[str, str] = field(default_factory=dict)
    file_overrides: list[dict[str, str]] = field(default_factory=list)
    bindings: list[ObjectBinding] = field(default_factory=list)
    verify: dict[str, Any] | None = None
    network_policy: SandboxNetworkPolicy = field(default_factory=SandboxNetworkPolicy)
    extra_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class SidecarSpec:
    """Optional sidecar sandbox attached to an agent."""

    name: str
    image: str
    provider: str = "akernel"
    port: int = DEFAULT_PORT
    ports: list[int] = field(default_factory=list)
    command: str | None = None
    env_url_key: str = ""
    task_filter: str = ""
    health_path: str = ""
    health_timeout_seconds: float = 0.0
    inject_kubeconfig: bool = False
    kubeconfig: str = ""
    cpu_request: int = DEFAULT_CPU_REQUEST
    cpu_limit: int = 2000
    memory_request_mb: int = DEFAULT_MEMORY_REQUEST_MB
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB
    needs_provider: bool = False
    env_from_payload: dict[str, str] = field(default_factory=dict)
    digest: str = ""
    network_policy: SandboxNetworkPolicy = field(default_factory=SandboxNetworkPolicy)
    extra_config: dict[str, Any] | None = None
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the sidecar configuration as a catalog mapping."""
        data: dict[str, Any] = {
            "name": self.name,
            "image": self.image,
            "provider": self.provider,
            "port": self.port,
            "env_url_key": self.env_url_key,
            "cpu_request": self.cpu_request,
            "cpu_limit": self.cpu_limit,
            "memory_request_mb": self.memory_request_mb,
            "memory_limit_mb": self.memory_limit_mb,
        }
        if self.command:
            data["command"] = self.command
        if self.ports:
            data["ports"] = list(self.ports)
        if self.task_filter:
            data["task_filter"] = self.task_filter
        if self.health_path:
            data["health_path"] = self.health_path
        if self.health_timeout_seconds:
            data["health_timeout_seconds"] = self.health_timeout_seconds
        if self.inject_kubeconfig:
            data["inject_kubeconfig"] = True
        if self.kubeconfig:
            data["kubeconfig"] = self.kubeconfig
        if self.needs_provider:
            data["needs_provider"] = True
        if self.env_from_payload:
            data["env_from_payload"] = dict(self.env_from_payload)
        if self.digest:
            data["digest"] = self.digest
        if not self.network_policy.is_public:
            data["network_policy"] = self.network_policy.to_dict()
        if self.extra_config:
            data["extra_config"] = dict(self.extra_config)
        if self.env:
            data["env"] = dict(self.env)
        return data


@dataclass(frozen=True)
class AgentSpec:
    """Resolved agent definition before run-specific overrides."""

    name: str
    runtime: str = ""
    gateway: str = ""
    description: str = ""
    sandbox: SandboxSpec | None = None
    sidecars: list[SidecarSpec] = field(default_factory=list)
    task_timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS
    min_retry_wait_seconds: float = DEFAULT_MIN_RETRY_WAIT_SECONDS
    max_retry_wait_seconds: float = DEFAULT_MAX_RETRY_WAIT_SECONDS
    retry_wait_multiplier: float = DEFAULT_RETRY_WAIT_MULTIPLIER
    env: dict[str, str] = field(default_factory=dict)
    model_id: str = ""
    judge_model_id: str = ""
    agent_models: dict[str, str] = field(default_factory=dict)
    output_prefix: str = ""
    n_concurrent: int | None = None


def parse_agent_specs(raw_agents: dict[str, Any]) -> dict[str, AgentSpec]:
    """Parse the catalog's closed ``agents`` mapping."""
    agents: dict[str, AgentSpec] = {}
    for raw_name, raw_section in raw_agents.items():
        if not isinstance(raw_section, dict):
            raise ValueError(f"catalog agents.{raw_name} must be a mapping")
        name = str(raw_name)
        agents[name] = _parse_agent(name, raw_section)
    if not agents:
        raise ValueError("catalog agents must not be empty")
    return agents


def _parse_agent(name: str, section: dict[str, Any]) -> AgentSpec:
    _reject_unknown(
        section,
        {
            "description",
            "runtime",
            "gateway",
            "model_id",
            "judge_model_id",
            "agent_models",
            "task_timeout",
            "task_retry",
            "n_concurrent",
            "output_prefix",
            "env",
            "sandbox",
            "sidecars",
        },
        f"agents.{name}",
    )
    sandbox = _parse_sandbox(section.get("sandbox"))
    task_retry = _parse_task_retry(section.get("task_retry"))
    return AgentSpec(
        name=name,
        runtime=_required_string(section, "runtime", f"agents.{name}"),
        gateway=_required_string(section, "gateway", f"agents.{name}"),
        description=str(section.get("description", "")),
        sandbox=sandbox,
        sidecars=_parse_sidecars(section.get("sidecars")),
        task_timeout_seconds=float(section.get("task_timeout", DEFAULT_TASK_TIMEOUT_SECONDS)),
        min_retry_wait_seconds=task_retry["min_retry_wait_seconds"],
        max_retry_wait_seconds=task_retry["max_retry_wait_seconds"],
        retry_wait_multiplier=task_retry["retry_wait_multiplier"],
        env=_string_dict(section.get("env")),
        model_id=str(section.get("model_id", "")),
        judge_model_id=str(section.get("judge_model_id", "")),
        agent_models=_parse_agent_models(section.get("agent_models")),
        output_prefix=str(section.get("output_prefix", "")),
        n_concurrent=_parse_optional_positive_int(section.get("n_concurrent"), "n_concurrent"),
    )


def _parse_task_retry(raw: object) -> dict[str, float]:
    values = {
        "min_retry_wait_seconds": DEFAULT_MIN_RETRY_WAIT_SECONDS,
        "max_retry_wait_seconds": DEFAULT_MAX_RETRY_WAIT_SECONDS,
        "retry_wait_multiplier": DEFAULT_RETRY_WAIT_MULTIPLIER,
    }
    if not isinstance(raw, dict):
        raise ValueError("agent task_retry must be a mapping")
    _reject_unknown(
        raw,
        {"min_retry_wait_seconds", "max_retry_wait_seconds", "retry_wait_multiplier"},
        "agent task_retry",
    )
    values["min_retry_wait_seconds"] = float(raw.get("min_retry_wait_seconds", values["min_retry_wait_seconds"]))
    values["max_retry_wait_seconds"] = float(raw.get("max_retry_wait_seconds", values["max_retry_wait_seconds"]))
    values["retry_wait_multiplier"] = float(raw.get("retry_wait_multiplier", values["retry_wait_multiplier"]))
    return values


def _parse_sandbox(raw: object) -> SandboxSpec | None:
    if not isinstance(raw, dict) or "image" not in raw:
        raise ValueError("agent sandbox must be a mapping containing image")
    _reject_unknown(
        raw,
        {
            "provider",
            "image",
            "port",
            "command",
            "image_from_payload",
            "task_runner",
            "digest",
            "cpu_request",
            "cpu_limit",
            "memory_request",
            "memory_limit",
            "env",
            "file_overrides",
            "bindings",
            "verify",
            "network_policy",
            "extra_config",
        },
        "agent sandbox",
    )
    cpu_limit = int(raw.get("cpu_limit", DEFAULT_CPU_LIMIT))
    memory_limit_mb = int(raw.get("memory_limit", DEFAULT_MEMORY_LIMIT_MB))
    return SandboxSpec(
        image=str(raw["image"]),
        provider=str(raw.get("provider", "akernel")),
        port=int(raw.get("port", DEFAULT_PORT)),
        cpu_request=_request_or_default(raw, "cpu_request", DEFAULT_CPU_REQUEST, cpu_limit),
        cpu_limit=cpu_limit,
        memory_request_mb=_request_or_default(raw, "memory_request", DEFAULT_MEMORY_REQUEST_MB, memory_limit_mb),
        memory_limit_mb=memory_limit_mb,
        command=str(raw["command"]) if "command" in raw else None,
        task_runner=_parse_task_runner(raw.get("task_runner", "")),
        image_from_payload=str(raw.get("image_from_payload", "")),
        digest=str(raw.get("digest", "")),
        env=_string_dict(raw.get("env")),
        file_overrides=_parse_file_overrides(raw.get("file_overrides")),
        bindings=_parse_bindings(raw.get("bindings")),
        verify=raw.get("verify") if isinstance(raw.get("verify"), dict) else None,
        network_policy=sandbox_network_policy_from_config(raw),
        extra_config=raw.get("extra_config") if isinstance(raw.get("extra_config"), dict) else None,
    )


def _parse_sidecars(raw: object) -> list[SidecarSpec]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("agent sidecars must be a list")
    sidecars: list[SidecarSpec] = []
    allowed = {
        "name",
        "image",
        "provider",
        "port",
        "ports",
        "command",
        "env_url_key",
        "task_filter",
        "health_path",
        "health_timeout_seconds",
        "inject_kubeconfig",
        "kubeconfig",
        "cpu_request",
        "cpu_limit",
        "memory_request",
        "memory_limit",
        "needs_provider",
        "env_from_payload",
        "digest",
        "network_policy",
        "extra_config",
        "env",
    }
    for index, item in enumerate(raw):
        path = f"agent sidecars[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{path} must be a mapping")
        _reject_unknown(item, allowed, path)
        cpu_limit = int(item.get("cpu_limit", 2000))
        memory_limit_mb = int(item.get("memory_limit", DEFAULT_MEMORY_LIMIT_MB))
        sidecars.append(
            SidecarSpec(
                name=str(item.get("name", f"sidecar_{index}")),
                image=_required_string(item, "image", path),
                provider=str(item.get("provider", "akernel")),
                port=int(item.get("port", DEFAULT_PORT)),
                ports=_parse_int_list(item.get("ports")),
                command=str(item["command"]) if "command" in item else None,
                env_url_key=str(item.get("env_url_key", "")),
                task_filter=str(item.get("task_filter", "")),
                health_path=str(item.get("health_path", "")),
                health_timeout_seconds=float(item.get("health_timeout_seconds", 0.0)),
                inject_kubeconfig=bool(item.get("inject_kubeconfig", False)),
                kubeconfig=str(item.get("kubeconfig", "")),
                cpu_request=_request_or_default(item, "cpu_request", DEFAULT_CPU_REQUEST, cpu_limit),
                cpu_limit=cpu_limit,
                memory_request_mb=_request_or_default(
                    item, "memory_request", DEFAULT_MEMORY_REQUEST_MB, memory_limit_mb
                ),
                memory_limit_mb=memory_limit_mb,
                needs_provider=bool(item.get("needs_provider", False)),
                env_from_payload=_string_dict(item.get("env_from_payload")),
                digest=str(item.get("digest", "")),
                network_policy=sandbox_network_policy_from_config(item),
                extra_config=(item.get("extra_config") if isinstance(item.get("extra_config"), dict) else None),
                env=_string_dict(item.get("env")),
            )
        )
    return sidecars


def _parse_task_runner(raw: object) -> str | list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return str(raw) if raw else ""


def _parse_int_list(raw: object) -> list[int]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("ports must be a list")
    return [int(item) for item in raw]


def _parse_agent_models(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("agent_models must be a mapping")
    models: dict[str, str] = {}
    for raw_role, raw_model in raw.items():
        role = str(raw_role).strip()
        if not role:
            raise ValueError("agent_models contains an empty role")
        model = str(raw_model).strip()
        if not model:
            raise ValueError(f"agent_models.{role} must not be empty")
        models[role] = model
    if "main" not in models:
        raise ValueError("agent_models.main is required")
    return models


def _parse_file_overrides(raw: object) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    return [
        {"src": str(item["src"]), "dst": str(item["dst"])}
        for item in raw
        if isinstance(item, dict) and "src" in item and "dst" in item
    ]


def _parse_bindings(raw: object) -> list[ObjectBinding]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("agent sandbox bindings must be a list")
    bindings: list[ObjectBinding] = []
    for index, item in enumerate(raw):
        path = f"agent sandbox bindings[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{path} must be a mapping")
        _reject_unknown(item, {"target", "oss_uri", "mode", "is_file"}, path)
        target_path = _required_string(item, "target", path)
        object_uri = _required_string(item, "oss_uri", path)
        if not object_uri.startswith("oss://"):
            raise ValueError(f"{path}.oss_uri must use oss://")
        mode = str(item.get("mode", "r"))
        if mode not in {"r", "w"}:
            raise ValueError(f"{path}.mode must be r or w")
        is_file = item.get("is_file", False)
        if not isinstance(is_file, bool):
            raise ValueError(f"{path}.is_file must be a boolean")
        bindings.append(
            ObjectBinding(
                target_path=target_path,
                object_uri=object_uri,
                mode=mode,
                kind="file" if is_file else "directory",
            )
        )
    return bindings


def _parse_optional_positive_int(raw: object, field_name: str) -> int | None:
    if raw is None:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return value


def _string_dict(raw: object) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("environment must be a mapping")
    return {str(key): str(value) for key, value in raw.items()}


def _required_string(raw: dict[str, Any], key: str, path: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value


def _reject_unknown(raw: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{path} has unknown keys: " + ", ".join(sorted(map(str, unknown))))
