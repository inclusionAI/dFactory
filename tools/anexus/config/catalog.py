"""Catalog resource contracts and strict parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from anexus.config.agents import AgentSpec, parse_agent_specs
from anexus.config.environment import resolve_env_refs
from anexus.config.loader import ConfigLoadError
from anexus.tasks import TaskSpec


@dataclass(frozen=True)
class GatewayMirror:
    name: str
    models: tuple[str, ...]
    api_key: str
    openai_upstream: str
    anthropic_upstream: str

    @property
    def route_label(self) -> str:
        label = re.sub(r"[^a-zA-Z0-9.-]", "-", self.name).strip("-").lower()
        return re.sub(r"-{2,}", "-", label) or "upstream"

    @property
    def protocol(self) -> str | None:
        if self.openai_upstream:
            return "openai"
        if self.anthropic_upstream:
            return "anthropic"
        return None

    @property
    def upstream_url(self) -> str:
        return self.openai_upstream or self.anthropic_upstream


@dataclass(frozen=True)
class ClusterSpec:
    name: str
    region: str


@dataclass(frozen=True)
class TlsSpec:
    verify: bool


@dataclass(frozen=True)
class ControlPlaneSpec:
    gateway_address: str
    server_address: str
    internal_address: str
    token: str
    tls: TlsSpec


@dataclass(frozen=True)
class DataTransferSpec:
    driver: str
    cli_path: str


@dataclass(frozen=True)
class SandboxProviderSpec:
    name: str
    driver: str
    control_plane: ControlPlaneSpec
    data_transfer: DataTransferSpec


@dataclass(frozen=True)
class ObjectStoreSpec:
    name: str
    driver: str
    endpoint: str
    public_endpoint: str
    bucket: str
    prefix: str
    access_key_id: str
    access_key_secret: str
    sign_version: str
    no_progress: bool


@dataclass(frozen=True)
class GatewayUpstreamProxySpec:
    request_timeout_seconds: float
    event_buffer_size: int
    tls: TlsSpec


@dataclass(frozen=True)
class GatewayResourcesSpec:
    cpu_request: int
    cpu_limit: int
    memory_request: int
    memory_limit: int


@dataclass(frozen=True)
class GatewayHealthSpec:
    command: str
    success_text: str
    attempts: int
    interval_seconds: float
    command_timeout_seconds: float
    external_ready_timeout_seconds: float
    external_ready_interval_seconds: float


@dataclass(frozen=True)
class GatewayAdminRetrySpec:
    timeout_seconds: float
    transient_status_codes: frozenset[int]


@dataclass(frozen=True)
class GatewayPoolSpec:
    max_live_instances: int
    max_active_sessions_per_instance: int
    max_sessions_per_instance_lifetime: int
    prefetch_instances: int
    idle_timeout_seconds: int


@dataclass(frozen=True)
class GatewayTraceSpec:
    object_store: str
    viewer_url: str
    export_policy: str


@dataclass(frozen=True)
class GatewaySpec:
    name: str
    driver: str
    sandbox_provider: str
    image: str
    port: int
    ports: tuple[int, ...]
    tunnel_proxy_port: int
    detached: bool
    upstream_proxy: GatewayUpstreamProxySpec
    command: str
    resources: GatewayResourcesSpec
    health: GatewayHealthSpec
    admin_retry: GatewayAdminRetrySpec
    tls: TlsSpec
    pool: GatewayPoolSpec
    environment: dict[str, str]
    mirrors: tuple[GatewayMirror, ...]
    trace: GatewayTraceSpec


@dataclass(frozen=True)
class ModelEnvironmentSpec:
    variable: str
    alias_role: str


@dataclass(frozen=True)
class AgentRuntimeSpec:
    name: str
    driver: str
    protocol: str
    session_environment: dict[str, str]
    model_environment: dict[str, ModelEnvironmentSpec]
    environment: dict[str, str]


@dataclass(frozen=True)
class CatalogResources:
    sandbox_providers: dict[str, SandboxProviderSpec] = field(default_factory=dict)
    object_stores: dict[str, ObjectStoreSpec] = field(default_factory=dict)
    gateways: dict[str, GatewaySpec] = field(default_factory=dict)
    agent_runtimes: dict[str, AgentRuntimeSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class Catalog:
    version: str
    cluster: ClusterSpec
    resources: CatalogResources
    agents: dict[str, AgentSpec]

    def get(self, name: str) -> AgentSpec | None:
        return self.agents.get(name)

    def names(self) -> list[str]:
        return sorted(self.agents)


def parse_catalog(data: dict[str, Any]) -> Catalog:
    """Parse the closed catalog schema and validate every resource reference."""
    _reject_unknown(data, {"version", "cluster", "resources", "agents"}, "catalog")
    version = _required_string(data, "version", "catalog")
    cluster = _parse_cluster(_required_mapping(data, "cluster", "catalog"))
    resources = _parse_resources(_required_mapping(data, "resources", "catalog"))
    agents = parse_agent_specs(_required_mapping(data, "agents", "catalog"))
    catalog = Catalog(version=version, cluster=cluster, resources=resources, agents=agents)
    _validate_references(catalog)
    return catalog


def resolve_catalog_for_agent(
    data: dict[str, Any],
    *,
    agent_name: str,
    env: dict[str, str],
    source_uri: str,
    task_payloads: list[dict[str, Any]] | None = None,
) -> Catalog:
    """Resolve environment references only in one Agent's consumed resource graph."""
    structural = parse_catalog(data)
    agent = structural.get(agent_name)
    if agent is None:
        known = ", ".join(structural.names()) or "<none>"
        raise ConfigLoadError(source_uri, f"unknown agent {agent_name!r}; known agents: {known}")

    provider_names = {agent.sandbox.provider} if agent.sandbox is not None else set()
    provider_names.update(sidecar.provider for sidecar in agent.sidecars)
    verifier_provider = _verifier_provider(agent)
    if verifier_provider:
        provider_names.add(verifier_provider)
    gateway = structural.resources.gateways[agent.gateway]
    provider_names.add(gateway.sandbox_provider)

    object_store_names = {gateway.trace.object_store}
    if agent.sandbox is not None:
        for binding in agent.sandbox.bindings:
            bucket = _oss_bucket(binding.object_uri)
            if bucket:
                matches = [name for name, store in structural.resources.object_stores.items() if store.bucket == bucket]
                if len(matches) != 1:
                    raise ConfigLoadError(
                        source_uri,
                        f"agents.{agent.name}.sandbox binding bucket {bucket!r} must match "
                        "exactly one resources.object_stores entry",
                    )
                object_store_names.add(matches[0])
    for bucket in _task_object_store_buckets(
        task_payloads or [],
        agent_name=agent_name,
        source_uri=source_uri,
    ):
        matches = [name for name, store in structural.resources.object_stores.items() if store.bucket == bucket]
        if len(matches) != 1:
            raise ConfigLoadError(
                source_uri,
                f"task payload bucket {bucket!r} must match exactly one resources.object_stores entry",
            )
        object_store_names.add(matches[0])

    raw_resources = _required_mapping(data, "resources", "catalog")
    selected = {
        "version": data["version"],
        "cluster": data["cluster"],
        "resources": {
            "sandbox_providers": _select_named(raw_resources, "sandbox_providers", provider_names),
            "object_stores": _select_named(raw_resources, "object_stores", object_store_names),
            "gateways": _select_named(raw_resources, "gateways", {agent.gateway}),
            "agent_runtimes": _select_named(raw_resources, "agent_runtimes", {agent.runtime}),
        },
        "agents": {agent_name: _required_mapping(data, "agents", "catalog")[agent_name]},
    }
    resolved = resolve_env_refs(selected, env=env, source_uri=source_uri)
    assert isinstance(resolved, dict)
    return parse_catalog(resolved)


def resolve_catalog_for_agents(
    data: dict[str, Any],
    *,
    agent_names: set[str],
    env: dict[str, str],
    source_uri: str,
    task_payloads: list[dict[str, Any]] | None = None,
) -> Catalog:
    """Resolve and merge the resource graphs consumed by multiple Agents."""
    if not agent_names:
        raise ConfigLoadError(source_uri, "task source did not select any agents")

    resolved_catalogs = [
        resolve_catalog_for_agent(
            data,
            agent_name=agent_name,
            env=env,
            source_uri=source_uri,
            task_payloads=[
                payload for payload in task_payloads or [] if str(payload.get("agent_name", "")).strip() == agent_name
            ],
        )
        for agent_name in sorted(agent_names)
    ]
    first = resolved_catalogs[0]
    sandbox_providers: dict[str, SandboxProviderSpec] = {}
    object_stores: dict[str, ObjectStoreSpec] = {}
    gateways: dict[str, GatewaySpec] = {}
    agent_runtimes: dict[str, AgentRuntimeSpec] = {}
    agents: dict[str, AgentSpec] = {}
    for catalog in resolved_catalogs:
        sandbox_providers.update(catalog.resources.sandbox_providers)
        object_stores.update(catalog.resources.object_stores)
        gateways.update(catalog.resources.gateways)
        agent_runtimes.update(catalog.resources.agent_runtimes)
        agents.update(catalog.agents)
    return Catalog(
        version=first.version,
        cluster=first.cluster,
        resources=CatalogResources(
            sandbox_providers=sandbox_providers,
            object_stores=object_stores,
            gateways=gateways,
            agent_runtimes=agent_runtimes,
        ),
        agents=agents,
    )


def _task_object_store_buckets(
    payloads: list[dict[str, Any]],
    *,
    agent_name: str,
    source_uri: str,
) -> set[str]:
    buckets: set[str] = set()
    for payload in payloads:
        task = TaskSpec.from_payload(
            run_id="catalog-resolution",
            agent_name=agent_name,
            payload=payload,
            source_uri=source_uri,
        )
        uris = [
            *(fixture.source_uri for fixture in task.fixtures.fixtures),
            *(artifact.uri for artifact in task.artifacts),
        ]
        buckets.update(bucket for uri in uris if (bucket := _oss_bucket(uri)))
    return buckets


def _parse_cluster(raw: dict[str, Any]) -> ClusterSpec:
    path = "cluster"
    _reject_unknown(raw, {"name", "region"}, path)
    return ClusterSpec(
        name=_required_string(raw, "name", path),
        region=_required_string(raw, "region", path),
    )


def _parse_resources(raw: dict[str, Any]) -> CatalogResources:
    path = "resources"
    keys = {"sandbox_providers", "object_stores", "gateways", "agent_runtimes"}
    _reject_unknown(raw, keys, path)
    for key in keys:
        if key not in raw:
            raise ValueError(f"{path}.{key} is required")
    return CatalogResources(
        sandbox_providers=_parse_named(
            _required_mapping(raw, "sandbox_providers", path),
            _parse_sandbox_provider,
            f"{path}.sandbox_providers",
        ),
        object_stores=_parse_named(
            _required_mapping(raw, "object_stores", path),
            _parse_object_store,
            f"{path}.object_stores",
        ),
        gateways=_parse_named(
            _required_mapping(raw, "gateways", path),
            _parse_gateway,
            f"{path}.gateways",
        ),
        agent_runtimes=_parse_named(
            _required_mapping(raw, "agent_runtimes", path),
            _parse_agent_runtime,
            f"{path}.agent_runtimes",
        ),
    )


def _parse_sandbox_provider(name: str, raw: dict[str, Any], path: str) -> SandboxProviderSpec:
    _reject_unknown(raw, {"driver", "control_plane", "data_transfer"}, path)
    driver = _required_string(raw, "driver", path)
    if driver != "akernel":
        raise ValueError(f"{path}.driver must be 'akernel'")
    control_raw = _required_mapping(raw, "control_plane", path)
    control_path = f"{path}.control_plane"
    _reject_unknown(
        control_raw,
        {"gateway_address", "server_address", "internal_address", "token", "tls"},
        control_path,
    )
    transfer_raw = _required_mapping(raw, "data_transfer", path)
    transfer_path = f"{path}.data_transfer"
    _reject_unknown(transfer_raw, {"driver", "cli_path"}, transfer_path)
    transfer_driver = _required_string(transfer_raw, "driver", transfer_path)
    if transfer_driver != "aliyun-oss":
        raise ValueError(f"{transfer_path}.driver must be 'aliyun-oss'")
    return SandboxProviderSpec(
        name=name,
        driver=driver,
        control_plane=ControlPlaneSpec(
            gateway_address=_required_string(control_raw, "gateway_address", control_path),
            server_address=_required_string(control_raw, "server_address", control_path),
            internal_address=_string(control_raw.get("internal_address"), f"{control_path}.internal_address"),
            token=_required_string(control_raw, "token", control_path),
            tls=_parse_tls(_required_mapping(control_raw, "tls", control_path), f"{control_path}.tls"),
        ),
        data_transfer=DataTransferSpec(
            driver=transfer_driver,
            cli_path=_required_string(transfer_raw, "cli_path", transfer_path),
        ),
    )


def _parse_object_store(name: str, raw: dict[str, Any], path: str) -> ObjectStoreSpec:
    keys = {
        "driver",
        "endpoint",
        "public_endpoint",
        "bucket",
        "prefix",
        "access_key_id",
        "access_key_secret",
        "sign_version",
        "no_progress",
    }
    _reject_unknown(raw, keys, path)
    driver = _required_string(raw, "driver", path)
    if driver != "aliyun-oss":
        raise ValueError(f"{path}.driver must be 'aliyun-oss'")
    return ObjectStoreSpec(
        name=name,
        driver=driver,
        endpoint=_required_string(raw, "endpoint", path),
        public_endpoint=_required_string(raw, "public_endpoint", path),
        bucket=_required_string(raw, "bucket", path),
        prefix=_required_string(raw, "prefix", path),
        access_key_id=_required_string(raw, "access_key_id", path),
        access_key_secret=_required_string(raw, "access_key_secret", path),
        sign_version=_required_string(raw, "sign_version", path),
        no_progress=_required_bool(raw, "no_progress", path),
    )


def _parse_gateway(name: str, raw: dict[str, Any], path: str) -> GatewaySpec:
    keys = {
        "driver",
        "sandbox_provider",
        "image",
        "port",
        "ports",
        "tunnel_proxy_port",
        "detached",
        "upstream_proxy",
        "command",
        "resources",
        "health",
        "admin_retry",
        "tls",
        "pool",
        "environment",
        "mirrors",
        "trace",
    }
    _reject_unknown(raw, keys, path)
    driver = _required_string(raw, "driver", path)
    if driver != "kgb":
        raise ValueError(f"{path}.driver must be 'kgb'")
    resources = _parse_gateway_resources(_required_mapping(raw, "resources", path), f"{path}.resources")
    ports = tuple(_required_int_list(raw, "ports", path))
    port = _positive_int(raw.get("port"), f"{path}.port")
    if port not in ports:
        raise ValueError(f"{path}.ports must include {path}.port")
    return GatewaySpec(
        name=name,
        driver=driver,
        sandbox_provider=_required_string(raw, "sandbox_provider", path),
        image=_required_string(raw, "image", path),
        port=port,
        ports=ports,
        tunnel_proxy_port=_positive_int(raw.get("tunnel_proxy_port"), f"{path}.tunnel_proxy_port"),
        detached=_required_bool(raw, "detached", path),
        upstream_proxy=_parse_gateway_upstream_proxy(
            _required_mapping(raw, "upstream_proxy", path), f"{path}.upstream_proxy"
        ),
        command=_required_string(raw, "command", path),
        resources=resources,
        health=_parse_gateway_health(_required_mapping(raw, "health", path), f"{path}.health"),
        admin_retry=_parse_gateway_admin_retry(_required_mapping(raw, "admin_retry", path), f"{path}.admin_retry"),
        tls=_parse_tls(_required_mapping(raw, "tls", path), f"{path}.tls"),
        pool=_parse_gateway_pool(_required_mapping(raw, "pool", path), f"{path}.pool"),
        environment=_required_string_mapping(raw, "environment", path),
        mirrors=tuple(_parse_mirrors(_required_mapping(raw, "mirrors", path), f"{path}.mirrors")),
        trace=_parse_gateway_trace(_required_mapping(raw, "trace", path), f"{path}.trace"),
    )


def _parse_gateway_upstream_proxy(raw: dict[str, Any], path: str) -> GatewayUpstreamProxySpec:
    _reject_unknown(raw, {"request_timeout_seconds", "event_buffer_size", "tls"}, path)
    return GatewayUpstreamProxySpec(
        request_timeout_seconds=_positive_float(raw.get("request_timeout_seconds"), f"{path}.request_timeout_seconds"),
        event_buffer_size=_positive_int(raw.get("event_buffer_size"), f"{path}.event_buffer_size"),
        tls=_parse_tls(_required_mapping(raw, "tls", path), f"{path}.tls"),
    )


def _parse_gateway_resources(raw: dict[str, Any], path: str) -> GatewayResourcesSpec:
    keys = {"cpu_request", "cpu_limit", "memory_request", "memory_limit"}
    _reject_unknown(raw, keys, path)
    resources = GatewayResourcesSpec(
        cpu_request=_positive_int(raw.get("cpu_request"), f"{path}.cpu_request"),
        cpu_limit=_positive_int(raw.get("cpu_limit"), f"{path}.cpu_limit"),
        memory_request=_positive_int(raw.get("memory_request"), f"{path}.memory_request"),
        memory_limit=_positive_int(raw.get("memory_limit"), f"{path}.memory_limit"),
    )
    if resources.cpu_request > resources.cpu_limit:
        raise ValueError(f"{path}.cpu_request must not exceed cpu_limit")
    if resources.memory_request > resources.memory_limit:
        raise ValueError(f"{path}.memory_request must not exceed memory_limit")
    return resources


def _parse_gateway_health(raw: dict[str, Any], path: str) -> GatewayHealthSpec:
    keys = {
        "command",
        "success_text",
        "attempts",
        "interval_seconds",
        "command_timeout_seconds",
        "external_ready_timeout_seconds",
        "external_ready_interval_seconds",
    }
    _reject_unknown(raw, keys, path)
    return GatewayHealthSpec(
        command=_required_string(raw, "command", path),
        success_text=_required_string(raw, "success_text", path),
        attempts=_positive_int(raw.get("attempts"), f"{path}.attempts"),
        interval_seconds=_positive_float(raw.get("interval_seconds"), f"{path}.interval_seconds"),
        command_timeout_seconds=_positive_float(raw.get("command_timeout_seconds"), f"{path}.command_timeout_seconds"),
        external_ready_timeout_seconds=_positive_float(
            raw.get("external_ready_timeout_seconds"), f"{path}.external_ready_timeout_seconds"
        ),
        external_ready_interval_seconds=_positive_float(
            raw.get("external_ready_interval_seconds"), f"{path}.external_ready_interval_seconds"
        ),
    )


def _parse_gateway_admin_retry(raw: dict[str, Any], path: str) -> GatewayAdminRetrySpec:
    _reject_unknown(raw, {"timeout_seconds", "transient_status_codes"}, path)
    codes = frozenset(_required_int_list(raw, "transient_status_codes", path))
    if not codes or any(code < 100 or code > 599 for code in codes):
        raise ValueError(f"{path}.transient_status_codes must contain HTTP status codes")
    return GatewayAdminRetrySpec(
        timeout_seconds=_positive_float(raw.get("timeout_seconds"), f"{path}.timeout_seconds"),
        transient_status_codes=codes,
    )


def _parse_gateway_pool(raw: dict[str, Any], path: str) -> GatewayPoolSpec:
    keys = {
        "max_live_instances",
        "max_active_sessions_per_instance",
        "max_sessions_per_instance_lifetime",
        "prefetch_instances",
        "idle_timeout_seconds",
    }
    _reject_unknown(raw, keys, path)
    prefetch = _non_negative_int(raw.get("prefetch_instances"), f"{path}.prefetch_instances")
    return GatewayPoolSpec(
        max_live_instances=_positive_int(raw.get("max_live_instances"), f"{path}.max_live_instances"),
        max_active_sessions_per_instance=_positive_int(
            raw.get("max_active_sessions_per_instance"), f"{path}.max_active_sessions_per_instance"
        ),
        max_sessions_per_instance_lifetime=_positive_int(
            raw.get("max_sessions_per_instance_lifetime"), f"{path}.max_sessions_per_instance_lifetime"
        ),
        prefetch_instances=prefetch,
        idle_timeout_seconds=_positive_int(raw.get("idle_timeout_seconds"), f"{path}.idle_timeout_seconds"),
    )


def _parse_gateway_trace(raw: dict[str, Any], path: str) -> GatewayTraceSpec:
    _reject_unknown(raw, {"object_store", "viewer_url", "export_policy"}, path)
    policy = _required_string(raw, "export_policy", path)
    if policy not in {"none", "on_release", "on_retire"}:
        raise ValueError(f"{path}.export_policy must be none, on_release, or on_retire")
    return GatewayTraceSpec(
        object_store=_required_string(raw, "object_store", path),
        viewer_url=_string(raw.get("viewer_url"), f"{path}.viewer_url"),
        export_policy=policy,
    )


def _parse_agent_runtime(name: str, raw: dict[str, Any], path: str) -> AgentRuntimeSpec:
    keys = {"driver", "protocol", "session_environment", "model_environment", "environment"}
    _reject_unknown(raw, keys, path)
    protocol = _required_string(raw, "protocol", path)
    if protocol not in {"openai", "anthropic"}:
        raise ValueError(f"{path}.protocol must be 'openai' or 'anthropic'")
    model_raw = _required_mapping(raw, "model_environment", path)
    models: dict[str, ModelEnvironmentSpec] = {}
    for raw_role, raw_value in model_raw.items():
        role = str(raw_role).strip()
        role_path = f"{path}.model_environment.{role}"
        if not role:
            raise ValueError(f"{path}.model_environment contains an empty role")
        value = _mapping(raw_value, role_path)
        _reject_unknown(value, {"variable", "alias_role"}, role_path)
        models[role] = ModelEnvironmentSpec(
            variable=_required_string(value, "variable", role_path),
            alias_role=_required_string(value, "alias_role", role_path),
        )
    if "main" not in models:
        raise ValueError(f"{path}.model_environment.main is required")
    session_environment = _required_string_mapping(raw, "session_environment", path)
    allowed_session_keys = {"base_url", "api_key", "model_id", "judge_model_id"}
    unknown_session_keys = set(session_environment) - allowed_session_keys
    if unknown_session_keys:
        raise ValueError(f"{path}.session_environment has unknown keys: " + ", ".join(sorted(unknown_session_keys)))
    missing_session_keys = {"base_url", "api_key"} - set(session_environment)
    if missing_session_keys:
        raise ValueError(
            f"{path}.session_environment is missing required keys: " + ", ".join(sorted(missing_session_keys))
        )
    return AgentRuntimeSpec(
        name=name,
        driver=_required_string(raw, "driver", path),
        protocol=protocol,
        session_environment=session_environment,
        model_environment=models,
        environment=_required_string_mapping(raw, "environment", path),
    )


def _parse_mirrors(raw: dict[str, Any], path: str) -> list[GatewayMirror]:
    mirrors: list[GatewayMirror] = []
    for raw_name, raw_value in raw.items():
        name = str(raw_name).strip()
        mirror_path = f"{path}.{name}"
        if not name:
            raise ValueError(f"{path} contains an empty mirror name")
        value = _mapping(raw_value, mirror_path)
        _reject_unknown(value, {"api_key", "openai_upstream", "anthropic_upstream", "models"}, mirror_path)
        openai = _string(value.get("openai_upstream"), f"{mirror_path}.openai_upstream").rstrip("/")
        anthropic = _string(value.get("anthropic_upstream"), f"{mirror_path}.anthropic_upstream").rstrip("/")
        if anthropic and not anthropic.endswith("/v1"):
            anthropic = f"{anthropic}/v1"
        if not openai and not anthropic:
            raise ValueError(f"{mirror_path} must define an upstream")
        raw_models = value.get("models")
        if not isinstance(raw_models, list) or not raw_models:
            raise ValueError(f"{mirror_path}.models must be a non-empty list")
        models = tuple(_string(item, f"{mirror_path}.models").strip() for item in raw_models)
        if any(not model for model in models):
            raise ValueError(f"{mirror_path}.models must not contain empty values")
        mirrors.append(
            GatewayMirror(
                name=name,
                models=models,
                api_key=_string(value.get("api_key"), f"{mirror_path}.api_key"),
                openai_upstream=openai,
                anthropic_upstream=anthropic,
            )
        )
    if not mirrors:
        raise ValueError(f"{path} must contain at least one mirror")
    return mirrors


def _parse_tls(raw: dict[str, Any], path: str) -> TlsSpec:
    _reject_unknown(raw, {"verify"}, path)
    return TlsSpec(verify=_required_bool(raw, "verify", path))


def _validate_references(catalog: Catalog) -> None:
    resources = catalog.resources
    for gateway_name, gateway in resources.gateways.items():
        if gateway.sandbox_provider not in resources.sandbox_providers:
            raise ValueError(
                f"resources.gateways.{gateway_name}.sandbox_provider references unknown "
                f"sandbox provider {gateway.sandbox_provider!r}"
            )
        if gateway.trace.object_store not in resources.object_stores:
            raise ValueError(
                f"resources.gateways.{gateway_name}.trace.object_store references unknown "
                f"object store {gateway.trace.object_store!r}"
            )
    for name, agent in catalog.agents.items():
        if agent.runtime not in resources.agent_runtimes:
            raise ValueError(f"agents.{name}.runtime references unknown runtime {agent.runtime!r}")
        if agent.gateway not in resources.gateways:
            raise ValueError(f"agents.{name}.gateway references unknown gateway {agent.gateway!r}")
        if agent.sandbox is None:
            raise ValueError(f"agents.{name}.sandbox is required")
        if agent.sandbox.provider not in resources.sandbox_providers:
            raise ValueError(
                f"agents.{name}.sandbox.provider references unknown sandbox provider {agent.sandbox.provider!r}"
            )
        for sidecar in agent.sidecars:
            if sidecar.provider not in resources.sandbox_providers:
                raise ValueError(
                    f"agents.{name}.sidecars provider references unknown sandbox provider {sidecar.provider!r}"
                )
        verifier_provider = _verifier_provider(agent)
        if verifier_provider and verifier_provider not in resources.sandbox_providers:
            raise ValueError(
                f"agents.{name}.sandbox.verify.sandbox.provider references unknown "
                f"sandbox provider {verifier_provider!r}"
            )
        for binding in agent.sandbox.bindings:
            bucket = _oss_bucket(binding.object_uri)
            matches = [store_name for store_name, store in resources.object_stores.items() if store.bucket == bucket]
            if len(matches) != 1:
                raise ValueError(
                    f"agents.{name}.sandbox binding bucket {bucket!r} must match "
                    "exactly one resources.object_stores entry"
                )
        runtime = resources.agent_runtimes[agent.runtime]
        unknown_roles = set(agent.agent_models) - set(runtime.model_environment)
        if unknown_roles:
            raise ValueError(
                f"agents.{name}.agent_models contains roles absent from "
                f"resources.agent_runtimes.{agent.runtime}.model_environment: " + ", ".join(sorted(unknown_roles))
            )


def _verifier_provider(agent: AgentSpec) -> str:
    if agent.sandbox is None or not isinstance(agent.sandbox.verify, dict):
        return ""
    raw_sandbox = agent.sandbox.verify.get("sandbox")
    if not isinstance(raw_sandbox, dict):
        return ""
    return str(raw_sandbox.get("provider") or agent.sandbox.provider).strip()


def _oss_bucket(uri: str) -> str:
    parts = urlsplit(uri)
    return parts.netloc if parts.scheme == "oss" else ""


def _select_named(raw_resources: dict[str, Any], key: str, names: set[str]) -> dict[str, Any]:
    mapping = _required_mapping(raw_resources, key, "resources")
    return {name: mapping[name] for name in names}


def _parse_named(raw: dict[str, Any], parser: Any, path: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw_name, raw_value in raw.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{path} contains an empty resource name")
        item_path = f"{path}.{name}"
        parsed[name] = parser(name, _mapping(raw_value, item_path), item_path)
    if not parsed:
        raise ValueError(f"{path} must not be empty")
    return parsed


def _reject_unknown(raw: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{path} has unknown keys: " + ", ".join(sorted(map(str, unknown))))


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def _required_mapping(raw: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    if key not in raw:
        raise ValueError(f"{path}.{key} is required")
    return _mapping(raw[key], f"{path}.{key}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    return value


def _required_string(raw: dict[str, Any], key: str, path: str) -> str:
    if key not in raw:
        raise ValueError(f"{path}.{key} is required")
    value = _string(raw[key], f"{path}.{key}")
    if not value:
        raise ValueError(f"{path}.{key} must not be empty")
    return value


def _required_string_mapping(raw: dict[str, Any], key: str, path: str) -> dict[str, str]:
    mapping = _required_mapping(raw, key, path)
    return {str(item_key): _string(item_value, f"{path}.{key}.{item_key}") for item_key, item_value in mapping.items()}


def _required_bool(raw: dict[str, Any], key: str, path: str) -> bool:
    if key not in raw or not isinstance(raw[key], bool):
        raise ValueError(f"{path}.{key} must be a boolean")
    return raw[key]


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be an integer >= 1")
    return value


def _non_negative_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be an integer >= 0")
    return value


def _positive_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{path} must be a number > 0")
    return float(value)


def _required_int_list(raw: dict[str, Any], key: str, path: str) -> list[int]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty list")
    return [_positive_int(item, f"{path}.{key}") for item in value]
