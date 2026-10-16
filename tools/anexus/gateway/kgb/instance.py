"""KGB sandbox lifecycle management using the sandbox provider protocol.

KGB is a sandbox-hosted LLM gateway service. The gateway module owns KGB
business lifecycle, but sandbox allocation and command execution go through the
existing `SandboxProvider` contract.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import ssl
import time
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha1
from time import monotonic
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from anexus.config.catalog import (
    AgentRuntimeSpec,
    GatewayPoolSpec,
    GatewaySpec,
    ObjectStoreSpec,
    SandboxProviderSpec,
)
from anexus.core.context import TaskContext
from anexus.gateway.kgb.trace_export import install_trace_export_command
from anexus.gateway.routing import (
    alias_target_model,
    build_mirror_routes,
    model_alias,
)
from anexus.gateway.upstream import (
    UPSTREAM_ROUTE_HEADER,
    GatewayUpstreamProxy,
    start_gateway_upstream_proxy,
)
from anexus.sandbox.provider import SandboxLaunchSpec, SandboxLease, SandboxProvider

logger = logging.getLogger(__name__)


@dataclass
class KGBConfig:
    """Configuration for one KGB sandbox."""

    gateway: GatewaySpec
    agent_runtime: AgentRuntimeSpec
    object_store: ObjectStoreSpec
    sandbox_provider: SandboxProviderSpec
    pool: GatewayPoolSpec
    judge_model_id: str
    agent_models: dict[str, str]
    trace_prefix: str


class KGBInstance:
    """Manage one KGB sandbox through `SandboxProvider`."""

    def __init__(
        self,
        config: KGBConfig,
        *,
        sandbox_provider: SandboxProvider | None = None,
    ) -> None:
        """Initialize a KGB instance."""
        self.config = config
        self._sandbox_provider = sandbox_provider
        self._lease: SandboxLease | None = None
        self._sandbox_id: str = ""
        self._url: str = ""
        self._internal_url: str = ""
        self._upstream_proxy: GatewayUpstreamProxy | None = None
        self._export_lock = asyncio.Lock()
        self._trace_files: list[str] = []
        self._task_api_key_names: dict[str, list[str]] = {}
        self._task_trace_refs: dict[str, list[dict[str, str]]] = {}
        self._api_key_ids: dict[str, int] = {}

    @property
    def lease(self) -> SandboxLease:
        """Return the provider lease for this KGB sandbox."""
        if self._lease is None:
            raise RuntimeError("KGB not started")
        return self._lease

    @property
    def url(self) -> str:
        """Return the external KGB URL."""
        if not self._url:
            raise RuntimeError("KGB not started")
        return self._url

    @property
    def sandbox_id(self) -> str:
        """Return the provider sandbox identifier."""
        return self._sandbox_id

    async def start(self, context: TaskContext | None = None) -> None:
        """Create the KGB sandbox and wait until it is healthy."""
        provider = self._require_sandbox_provider()
        routes = build_mirror_routes(self.config.gateway.mirrors)
        if not routes:
            raise RuntimeError("KGB requires at least one mirror with an upstream URL")

        provider_context = _make_kgb_provider_context(context or _default_kgb_context())
        launch = self._launch_spec(routes=routes)
        try:
            self._lease = await provider.create(provider_context, launch)
            self._sandbox_id = self._lease.sandbox_id
            self._url = (await provider.port_url(self._lease, self.config.gateway.port, internal=False)).rstrip("/")
            self._internal_url = (await provider.port_url(self._lease, self.config.gateway.port, internal=True)).rstrip(
                "/"
            )
            logger.info(
                "KGB sandbox created: id=%s url=%s upstream_proxy=%s",
                self._sandbox_id,
                self._url,
                self._upstream_proxy.tunnel_upstream if self._upstream_proxy else "",
            )
            await self._wait_until_healthy()
            await self._wait_until_external_admin_ready()
            await self._install_trace_exporter_patch()
        except BaseException as exc:
            try:
                await asyncio.shield(self.stop())
            except Exception as cleanup_exc:
                exc.add_note(f"KGB start cleanup also failed: {cleanup_exc}")
            raise

    def configure_providers(self) -> None:
        """Register upstream providers and model aliases on KGB."""
        all_models = self._registered_models()
        upstream_specs = self._build_upstream_specs()
        if not upstream_specs:
            raise RuntimeError("No valid upstream specs for KGB providers")

        for model in all_models:
            alias_prefs = self._model_alias_preferences(model)
            for alias_name in alias_prefs:
                self._admin_post(
                    "/api/admin/models",
                    {
                        "requested_model": alias_name,
                        "strategy": "priority",
                    },
                )

        provider_ids: list[tuple[dict[str, Any], int]] = []
        for spec in upstream_specs:
            upstream_path = urlparse(str(spec["url"])).path.rstrip("/") or ""
            provider_base_url = f"http://127.0.0.1:{self.config.gateway.tunnel_proxy_port}{upstream_path}"
            extra_headers = {UPSTREAM_ROUTE_HEADER: str(spec["route_key"])}
            resp = self._admin_post(
                "/api/admin/providers",
                {
                    "name": spec["name"],
                    "base_url": provider_base_url,
                    "protocol": spec["protocol"],
                    "api_key": spec.get("api_key") or "dummy",
                    "extra_headers": extra_headers,
                },
            )
            provider_ids.append((spec, resp.get("id", 0)))

        for model in all_models:
            upstream_name = alias_target_model(model)
            alias_prefs = self._model_alias_preferences(model)
            eligible = [
                (s, pid)
                for s, pid in provider_ids
                if model in {alias_target_model(str(candidate)) for candidate in s.get("models", ())}
            ]
            if not eligible:
                eligible = provider_ids
            enable_protocol_preference = len({str(spec.get("kind")) for spec, _provider_id in eligible}) > 1

            for spec, provider_id in eligible:
                for alias_name, preferred_kind in alias_prefs.items():
                    mapping: dict[str, Any] = {
                        "requested_model": alias_name,
                        "provider_id": provider_id,
                        "target_model_name": upstream_name,
                        "billing_mode": "inherit_model_default",
                    }
                    if enable_protocol_preference:
                        mapping["priority"] = 0 if str(spec.get("kind")) == preferred_kind else 100
                    self._admin_post("/api/admin/model-providers", mapping)

        logger.info(
            "KGB configured: %d models, %d providers, %d specs",
            len(all_models),
            len(provider_ids),
            len(upstream_specs),
        )

    def build_agent_env(
        self,
        agent_name: str,
        task_id: str,
        *,
        api_key: str | None = None,
    ) -> dict[str, str]:
        """Build LLM environment variables for an agent sandbox."""
        key = api_key or self.create_api_key(agent_name, task_id, role="worker")
        agent_models = self._resolved_agent_models()
        runtime = self.config.agent_runtime
        session_env = runtime.session_environment
        env: dict[str, str] = {
            session_env["base_url"]: self._internal_url,
            session_env["api_key"]: key,
        }
        if model_id_variable := session_env.get("model_id"):
            env[model_id_variable] = model_alias(runtime.protocol, agent_models["main"], "")
        for role, model in agent_models.items():
            model_environment = runtime.model_environment[role]
            env[model_environment.variable] = model_alias(
                runtime.protocol,
                model,
                "",
                role=model_environment.alias_role,
            )
        if self.config.judge_model_id and (judge_model_variable := session_env.get("judge_model_id")):
            env[judge_model_variable] = model_alias(
                runtime.protocol,
                alias_target_model(self.config.judge_model_id),
                "",
            )
        return env

    def _resolved_agent_models(self) -> dict[str, str]:
        """Return the model roles explicitly selected by the agent."""
        return {role: alias_target_model(model) for role, model in self.config.agent_models.items()}

    def _registered_models(self) -> list[str]:
        """Return all selectable and advertised upstream models once."""
        candidates = [
            *self._resolved_agent_models().values(),
            self.config.judge_model_id,
            *(model for mirror in self.config.gateway.mirrors for model in mirror.models),
        ]
        return list(
            dict.fromkeys(model for candidate in candidates if (model := alias_target_model(str(candidate).strip())))
        )

    def create_api_key(
        self,
        agent_name: str,
        task_id: str,
        role: str = "worker",
        *,
        identity: str = "",
    ) -> str:
        """Create a KGB API key for one agent/task role."""
        key_name = _make_kgb_key_name(agent_name, task_id, role, identity=identity)
        resp = self._admin_post("/api/admin/api-keys", {"key_name": key_name})
        key_value = resp.get("key_value", "")
        key_id = _optional_int(resp.get("id") or resp.get("api_key_id"))
        if key_id is not None:
            self._api_key_ids[key_name] = key_id
        trace_file = f"{key_name}.jsonl"
        self._trace_files.append(trace_file)
        self._task_api_key_names.setdefault(task_id, []).append(key_name)
        self._task_trace_refs.setdefault(task_id, []).append(
            self._trace_ref(key_name=key_name, trace_file=trace_file, role=role)
        )
        logger.info("KGB API key created: %s", key_name)
        return str(key_value)

    def trace_refs_for_task(self, task_id: str) -> list[dict[str, str]]:
        """Return trace references expected for one task."""
        return [dict(ref) for ref in self._task_trace_refs.get(task_id, [])]

    def log_count(self, *, api_key_name: str = "", timeout: float = 5.0) -> int:
        """Return KGB admin log count."""
        if not self._url:
            return 0
        params: dict[str, object] = {"page_size": 5}
        if api_key_name:
            params["api_key_name"] = api_key_name
        resp = _try_http_get(
            f"{self._url}/api/admin/logs?{urlencode(params)}",
            timeout=timeout,
            tls_verify=self.config.gateway.tls.verify,
        )
        if resp is None:
            return 0
        return _safe_int(resp.get("total"))

    def log_counts_by_task(
        self,
        task_ids: Iterable[str],
        *,
        timeout: float = 5.0,
    ) -> dict[str, int]:
        """Return KGB admin log counts grouped by task."""
        counts: dict[str, int] = {}
        for task_id in task_ids:
            key_names = self._task_api_key_names.get(task_id, [])
            if not key_names:
                continue
            counts[task_id] = sum(self.log_count(api_key_name=key_name, timeout=timeout) for key_name in key_names)
        return counts

    def request_activity(
        self,
        api_key_name: str,
        *,
        timeout: float = 2.0,
    ) -> dict[str, Any]:
        """Return the latest request summary and total for one API key."""
        if not self._url:
            raise RuntimeError("KGB is not started")
        params: dict[str, object] = {
            "page_size": 1,
            "sort_by": "request_time",
            "sort_order": "desc",
        }
        api_key_id = self._api_key_ids.get(api_key_name)
        if api_key_id is not None:
            params["api_key_id"] = api_key_id
        else:
            params["api_key_name"] = api_key_name
        response = _http_get(
            f"{self._url}/api/admin/logs?{urlencode(params)}",
            timeout=timeout,
            tls_verify=self.config.gateway.tls.verify,
        )
        items = response.get("items")
        latest = items[0] if isinstance(items, list) and items and isinstance(items[0], dict) else {}
        resolved_api_key_id = api_key_id or _optional_int(latest.get("api_key_id"))
        if resolved_api_key_id is not None:
            self._api_key_ids[api_key_name] = resolved_api_key_id
        return {
            "api_key_id": resolved_api_key_id,
            "api_key_name": api_key_name,
            "request_count": _safe_int(response.get("total")),
            "last_request_time": str(latest.get("request_time") or ""),
            "response_status": _optional_int(latest.get("response_status")),
            "first_byte_delay_ms": _optional_int(latest.get("first_byte_delay_ms")),
            "total_time_ms": _optional_int(latest.get("total_time_ms")),
            "retry_count": _safe_int(latest.get("retry_count")),
            "input_tokens": _optional_int(latest.get("input_tokens")),
            "output_tokens": _optional_int(latest.get("output_tokens")),
        }

    async def export_task_traces(self, trace_files: list[str] | None = None) -> list[str]:
        """Export and upload trace JSONL files from the KGB sandbox."""
        files = trace_files or self._trace_files
        if not files:
            return []

        provider = self._require_sandbox_provider()
        lease = self.lease
        async with self._export_lock:
            result = await provider.exec(
                lease,
                "python /opt/trace_export/trace_exporter_cron.py --once",
                timeout=60,
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    "KGB trace exporter failed: "
                    f"exit_code={result.exit_code} "
                    f"stdout={result.stdout} stderr={result.stderr}"
                )

            uploaded: list[str] = []
            for trace_file in files:
                if not trace_file.endswith(".jsonl"):
                    continue
                local = f"/opt/traces/{trace_file}"
                uri = f"oss://{self.config.object_store.bucket}/{self.config.trace_prefix}/traces/{trace_file}"
                local_q = shlex.quote(local)
                uri_q = shlex.quote(uri)
                result = await provider.exec(
                    lease,
                    f"if [ ! -s {local_q} ]; then "
                    f"echo TRACE_MISSING:{shlex.quote(trace_file)}; "
                    "ls -la /opt/traces 2>&1 || true; "
                    "exit 86; "
                    "fi; "
                    f"{shlex.quote(self.config.sandbox_provider.data_transfer.cli_path)} "
                    f"cp {local_q} {uri_q} -f "
                    f'--endpoint="${{OSS_ENDPOINT}}" --access-key-id="${{OSS_KEY_ID}}" '
                    f'--access-key-secret="${{OSS_KEY_SECRET}}" '
                    f"--sign-version {shlex.quote(self.config.object_store.sign_version)} "
                    f"{'--no-progress' if self.config.object_store.no_progress else ''}; "
                    f"{shlex.quote(self.config.sandbox_provider.data_transfer.cli_path)} "
                    f"stat {uri_q} "
                    f'--endpoint="${{OSS_ENDPOINT}}" --access-key-id="${{OSS_KEY_ID}}" '
                    f'--access-key-secret="${{OSS_KEY_SECRET}}" '
                    f"--sign-version {shlex.quote(self.config.object_store.sign_version)} "
                    ">/dev/null",
                    timeout=300,
                )
                if result.exit_code != 0:
                    raise RuntimeError(
                        "KGB trace upload failed: "
                        f"trace_file={trace_file} exit_code={result.exit_code} "
                        f"stdout={result.stdout} stderr={result.stderr}"
                    )
                uploaded.append(trace_file)
                logger.info("Trace uploaded: %s", trace_file)

            return uploaded

    def trace_viewer_url(self) -> str:
        """Generate the trace viewer URL for this KGB trace prefix."""
        cfg = self.config
        if not cfg.gateway.trace.viewer_url:
            return ""
        import base64

        viewer_config = json.dumps(
            {
                "endpoint": cfg.object_store.public_endpoint,
                "accessKeyId": cfg.object_store.access_key_id,
                "accessKeySecret": cfg.object_store.access_key_secret,
                "bucketName": cfg.object_store.bucket,
                "prefix": f"{cfg.trace_prefix}/traces",
            },
            separators=(",", ":"),
        )
        b64 = base64.b64encode(viewer_config.encode()).decode()
        return f"{cfg.gateway.trace.viewer_url.rstrip('/')}#/oss/{b64}"

    async def stop(self) -> None:
        """Delete the KGB sandbox and stop its host-side upstream proxy."""
        lease = self._lease
        provider = self._sandbox_provider
        errors: list[Exception] = []
        if lease is not None and provider is not None:
            try:
                await provider.delete(lease)
            except Exception as exc:
                errors.append(exc)
            else:
                self._lease = None
                self._url = ""
                self._internal_url = ""
        if self._upstream_proxy is not None:
            try:
                self._upstream_proxy.shutdown()
            except Exception as exc:
                errors.append(exc)
            else:
                self._upstream_proxy = None
        if errors:
            raise errors[0]

    def _launch_spec(
        self,
        *,
        routes: dict[str, str],
    ) -> SandboxLaunchSpec:
        """Build the provider-facing KGB launch spec."""
        env = {
            **self.config.gateway.environment,
            **_object_store_environment(self.config.object_store),
            "OSS_PREFIX": self.config.trace_prefix,
        }
        extra_config: dict[str, object] = {}
        self._upstream_proxy = start_gateway_upstream_proxy(
            routes,
            timeout_seconds=self.config.gateway.upstream_proxy.request_timeout_seconds,
            event_buffer_size=self.config.gateway.upstream_proxy.event_buffer_size,
            tls_verify=self.config.gateway.upstream_proxy.tls.verify,
        )
        extra_config["tunnel_upstream"] = self._upstream_proxy.tunnel_upstream
        extra_config["tunnel_proxy_port"] = self.config.gateway.tunnel_proxy_port
        extra_config["idle_timeout"] = self.config.pool.idle_timeout_seconds
        extra_config["detached"] = self.config.gateway.detached

        return SandboxLaunchSpec(
            image=self.config.gateway.image,
            provider=self.config.gateway.sandbox_provider,
            port=self.config.gateway.port,
            ports=list(self.config.gateway.ports),
            cpu_request=self.config.gateway.resources.cpu_request,
            cpu_limit=self.config.gateway.resources.cpu_limit,
            memory_request_mb=self.config.gateway.resources.memory_request,
            memory_limit_mb=self.config.gateway.resources.memory_limit,
            command=self.config.gateway.command,
            name="kgb",
            env=env,
            extra_config=extra_config or None,
        )

    async def _wait_until_healthy(self) -> None:
        """Poll KGB health from inside the sandbox."""
        provider = self._require_sandbox_provider()
        lease = self.lease
        health = self.config.gateway.health
        for i in range(health.attempts):
            result = await provider.exec(
                lease,
                health.command,
                timeout=health.command_timeout_seconds,
            )
            if health.success_text in result.stdout:
                logger.info("KGB healthy after %ds", i)
                return
            await asyncio.sleep(health.interval_seconds)
        log = await provider.exec(
            lease,
            "tail -50 /tmp/kgb_server.log 2>/dev/null || echo 'no log'",
            timeout=health.command_timeout_seconds,
        )
        raise RuntimeError(f"KGB failed health check after {health.attempts} attempts. Log:\n{log.stdout[:2000]}")

    async def _wait_until_external_admin_ready(self) -> None:
        """Wait for host-visible health and admin routes to become usable."""
        timeout = self.config.gateway.health.external_ready_timeout_seconds
        interval = self.config.gateway.health.external_ready_interval_seconds
        deadline = monotonic() + timeout
        last_error = ""
        while True:
            ready, status_code, error = await asyncio.to_thread(self._probe_external_admin)
            if ready:
                return
            last_error = error
            if status_code and status_code not in self.config.gateway.admin_retry.transient_status_codes:
                raise RuntimeError(f"KGB external admin readiness failed with HTTP {status_code}: {error}")
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"KGB external admin endpoint did not become ready within {timeout:.1f}s: {last_error}"
                )
            await asyncio.sleep(min(interval, remaining))

    def _probe_external_admin(self) -> tuple[bool, int, str]:
        """Probe host-visible KGB health and one read-only admin endpoint."""
        try:
            health = _http_get(
                f"{self._url}/health",
                timeout=self.config.gateway.health.command_timeout_seconds,
                tls_verify=self.config.gateway.tls.verify,
            )
            if health.get("status") != "healthy":
                return False, 0, f"unexpected health payload: {health!r}"
            _http_get(
                f"{self._url}/api/admin/logs?page_size=1",
                timeout=self.config.gateway.health.command_timeout_seconds,
                tls_verify=self.config.gateway.tls.verify,
            )
            return True, 200, ""
        except HTTPError as exc:
            return False, int(exc.code), str(exc)
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            return False, 0, str(exc) or exc.__class__.__name__

    def _build_upstream_specs(self) -> list[dict[str, Any]]:
        """Build one upstream spec per mirror/protocol pair."""
        specs: list[dict[str, Any]] = []
        for mirror in self.config.gateway.mirrors:
            if not mirror.models:
                continue
            for kind, upstream in (
                ("openai", mirror.openai_upstream),
                ("anthropic", mirror.anthropic_upstream),
            ):
                if not upstream:
                    continue
                name = f"{mirror.route_label}-{kind}"
                specs.append(
                    {
                        "kind": kind,
                        "name": name,
                        "protocol": kind,
                        "url": upstream.rstrip("/"),
                        "api_key": mirror.api_key,
                        "models": tuple(mirror.models),
                        "route_key": name,
                    }
                )
        return specs

    def _model_alias_preferences(self, model: str) -> dict[str, str]:
        """Return model aliases mapped to their preferred protocol kind."""
        runtime_protocol = self.config.agent_runtime.protocol
        names: dict[str, str] = {
            model_alias("openai", model, ""): "openai",
            model_alias("anthropic", model, ""): "anthropic",
        }
        for role in self._resolved_agent_models():
            model_environment = self.config.agent_runtime.model_environment[role]
            names[
                model_alias(
                    runtime_protocol,
                    model,
                    "",
                    role=model_environment.alias_role,
                )
            ] = runtime_protocol
        return names

    def _admin_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to KGB admin with bounded transient-route retries."""
        url = f"{self._url}{path}"
        retry = self.config.gateway.admin_retry
        deadline = monotonic() + retry.timeout_seconds
        while True:
            try:
                return _http_post(
                    url,
                    payload,
                    timeout=retry.timeout_seconds,
                    tls_verify=self.config.gateway.tls.verify,
                )
            except KGBAdminAPIError as exc:
                if exc.status_code not in retry.transient_status_codes:
                    raise
                last_error: Exception = exc
            except (URLError, TimeoutError, OSError) as exc:
                last_error = exc
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"KGB admin API did not become reachable within {retry.timeout_seconds:.1f}s: {last_error}"
                ) from last_error
            time.sleep(min(self.config.gateway.health.external_ready_interval_seconds, remaining))

    def _trace_ref(self, *, key_name: str, trace_file: str, role: str) -> dict[str, str]:
        """Build a JSON-serializable trace reference."""
        ref = {
            "api_key_name": key_name,
            "trace_file": trace_file,
            "role": role,
        }
        bucket = self.config.object_store.bucket
        if bucket and self.config.trace_prefix:
            ref["oss_uri"] = f"oss://{bucket}/{self.config.trace_prefix}/traces/{trace_file}"
        viewer_url = self.trace_viewer_url()
        if viewer_url:
            ref["trace_viewer_url"] = viewer_url
        return ref

    async def _install_trace_exporter_patch(self) -> None:
        """Install Anexus' patched trace exporter into the KGB sandbox."""
        result = await self._require_sandbox_provider().exec(
            self.lease,
            install_trace_export_command(),
            timeout=15,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                "failed to install patched KGB trace exporter: "
                f"stdout={result.stdout[-1000:]} stderr={result.stderr[-1000:]}"
            )
        logger.info("Patched KGB trace exporter installed: %s", result.stdout.strip())

    def _require_sandbox_provider(self) -> SandboxProvider:
        """Return the configured sandbox provider."""
        if self._sandbox_provider is None:
            raise RuntimeError("KGBInstance requires a SandboxProvider")
        return self._sandbox_provider


def _default_kgb_context() -> TaskContext:
    """Build a fallback context for direct KGB lifecycle tests."""
    return TaskContext(
        run_id="kgb",
        task_id="kgb",
        version=1,
        attempt_id="kgb",
        agent_name="kgb",
        agent_spec_name="kgb",
    )


def _make_kgb_provider_context(context: TaskContext) -> TaskContext:
    """Return an AKernel route-safe context for the KGB service sandbox."""

    safe_agent = re.sub(r"[^a-z0-9]+", "", context.agent_name.lower())[:8] or "agent"
    nonce = sha1(f"{context.run_id}:{context.task_id}:{context.attempt_id}:{uuid4().hex}".encode()).hexdigest()[:8]
    return TaskContext(
        run_id=context.run_id,
        task_id=context.task_id,
        version=context.version,
        attempt_id=nonce,
        agent_name=safe_agent,
        agent_spec_name=context.agent_spec_name,
    )


def _make_kgb_key_name(
    agent_name: str,
    task_id: str,
    role: str = "worker",
    *,
    identity: str = "",
) -> str:
    """Generate a KGB-compatible API key name."""
    raw = f"{agent_name}-{task_id}-{identity}-{role}"
    digest = sha1(raw.encode()).hexdigest()[:10]
    safe_base = re.sub(
        r"[^A-Za-z0-9_\- ]+",
        "-",
        f"{agent_name}-{task_id}-{identity}" if identity else f"{agent_name}-{task_id}",
    )
    safe_base = re.sub(r"[-_ ]{2,}", "-", safe_base).strip("-_ ")
    if not safe_base:
        safe_base = "kgb-key"
    suffix = f"-{role}-{digest}"
    budget = max(1, 100 - len(suffix))
    return f"{safe_base[:budget].rstrip('-_ ')}{suffix}"


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _http_get(url: str, *, timeout: float, tls_verify: bool) -> dict[str, Any]:
    """GET a JSON document."""
    req = Request(url, method="GET")
    ctx = _ssl_context(url, tls_verify=tls_verify)
    with urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode())


def _try_http_get(url: str, *, timeout: float, tls_verify: bool) -> dict[str, Any] | None:
    """GET a JSON document without raising transport errors."""
    req = Request(url, method="GET")
    ctx = _ssl_context(url, tls_verify=tls_verify)
    try:
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode())
    except (URLError, OSError, ValueError):
        return None


def _safe_int(value: object) -> int:
    """Coerce a value to an integer, returning zero on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class KGBAdminAPIError(RuntimeError):
    """Represent a non-success KGB admin HTTP response."""

    def __init__(self, message: str, *, status_code: int) -> None:
        """Initialize an admin response error."""
        super().__init__(message)
        self.status_code = status_code


def _http_post(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    tls_verify: bool,
) -> dict[str, Any]:
    """POST a JSON document and decode the JSON response."""
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    ctx = _ssl_context(url, tls_verify=tls_verify)
    try:
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            body = resp.read().decode()
            return json.loads(body) if body.strip() else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise KGBAdminAPIError(
            f"KGB admin API {url} returned {exc.code}: {body[:200]}",
            status_code=int(exc.code),
        ) from exc


def _ssl_context(url: str, *, tls_verify: bool) -> ssl.SSLContext | None:
    if not url.startswith("https") or tls_verify:
        return None
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _object_store_environment(store: ObjectStoreSpec) -> dict[str, str]:
    return {
        "OSS_ENDPOINT": store.endpoint,
        "OSS_KEY_ID": store.access_key_id,
        "OSS_KEY_SECRET": store.access_key_secret,
        "OSS_BUCKET": store.bucket,
    }
