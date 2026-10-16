"""Translate provider-neutral launch specs into AKernel inputs."""

from __future__ import annotations

import re
from hashlib import sha1
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from akernel_sdk import HttpReverseTunnel, NetworkPolicy

from anexus.core.context import TaskContext
from anexus.sandbox.network import (
    SandboxNetworkAllowlistEntryType,
    SandboxNetworkMode,
    SandboxNetworkPolicy,
    classify_sandbox_network_allowlist_entry,
)
from anexus.sandbox.provider import SandboxLaunchSpec


def build_sandbox_kwargs(
    context: TaskContext,
    spec: SandboxLaunchSpec,
) -> dict[str, Any]:
    """Build keyword arguments accepted by ``akernel_sdk.Sandbox``."""

    extra_config = dict(spec.extra_config or {})
    network_policy = apply_network_policy(extra_config, spec.network_policy)

    tunnel_upstream = extra_config.pop("tunnel_upstream", None)
    tunnel_proxy_port = int(extra_config.pop("tunnel_proxy_port", 0))
    idle_timeout = extra_config.pop("idle_timeout", None)
    detached = extra_config.pop("detached", None)

    kwargs: dict[str, Any] = {
        "image": spec.image,
        "cpu": spec.cpu_request,
        "cpu_limit": spec.cpu_limit,
        "memory": spec.memory_request_mb,
        "mem_limit": spec.memory_limit_mb,
        "port_forwardings": (list(spec.ports) if spec.ports else ([spec.port] if spec.port else [])),
        "name": make_akernel_sandbox_name(context, spec),
        "env": dict(spec.env) if spec.env else None,
        "extra_config": extra_config,
    }
    if network_policy is not None:
        kwargs["network_policy"] = network_policy
    if tunnel_upstream:
        tunnel_kwargs: dict[str, Any] = {"target": str(tunnel_upstream)}
        if tunnel_proxy_port:
            tunnel_kwargs.update(
                reverse_port=tunnel_proxy_port - 1,
                listen_port=tunnel_proxy_port,
            )
        kwargs["reverse_tunnel"] = HttpReverseTunnel(**tunnel_kwargs)
    if idle_timeout is not None:
        kwargs["idle_timeout"] = int(idle_timeout)
    if detached is not None:
        kwargs["detached"] = parse_bool(detached)
    return kwargs


def make_akernel_sandbox_name(
    context: TaskContext,
    spec: SandboxLaunchSpec,
    *,
    max_len: int = 55,
) -> str:
    """Build a route-safe ``<role>-<agent>-<hash>`` sandbox name."""

    role = _name_segment(_sandbox_role(spec), fallback="sandbox")
    agent = _name_segment(context.agent_name, fallback="agent")
    raw = f"{context.run_id}:{context.task_id}:{context.attempt_id}:{spec.name}"
    task_hash = sha1(raw.encode()).hexdigest()[:8]
    label = f"{role}-{agent}-{task_hash}"
    if len(label) <= max_len:
        return label
    return label[:max_len].rstrip("-") or task_hash


def apply_network_policy(
    extra_config: dict[str, Any],
    policy: SandboxNetworkPolicy,
) -> NetworkPolicy | None:
    """Translate supported policies to the public SDK's typed API.

    The public SDK does not yet model CIDR allowlists, so that supported
    ANexus mode remains a backend-owned ``extra_config`` extension.
    """

    if policy.network_mode == SandboxNetworkMode.PUBLIC:
        return None
    if policy.network_mode == SandboxNetworkMode.NO_NETWORK:
        extra_config.pop("blockNetwork", None)
        extra_config.pop("cidrAllowlist", None)
        return NetworkPolicy.block()
    if policy.network_mode == SandboxNetworkMode.ALLOWLIST:
        extra_config["blockNetwork"] = False
        extra_config["cidrAllowlist"] = ",".join(_cidr_allowlist(policy))
        return None
    return None


def file_override_paths(
    override: dict[str, str],
    *,
    index: int,
) -> tuple[Path, str]:
    """Validate one local-to-sandbox file override."""

    raw_source = str(override.get("src", "")).strip()
    target_path = str(override.get("dst", "")).strip()
    if not raw_source or not target_path:
        raise ValueError(f"file override {index} requires src and dst (src={raw_source!r}, dst={target_path!r})")
    parts = urlsplit(raw_source)
    if parts.scheme == "file":
        source_path = Path(unquote(parts.path)).expanduser()
    elif parts.scheme:
        raise ValueError(f"file override {index} source must be a local path: {raw_source}")
    else:
        source_path = Path(raw_source).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"file override {index} source is missing: {source_path}")
    return source_path, target_path


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _sandbox_role(spec: SandboxLaunchSpec) -> str:
    role = str(spec.name or "").strip()
    if role == "main":
        return "agent"
    if role in {"agent", "kgb", "verify"}:
        return role
    return "sidecar"


def _name_segment(value: str, *, fallback: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())[:8] or fallback[:8]


def _cidr_allowlist(policy: SandboxNetworkPolicy) -> list[str]:
    cidr_blocks: list[str] = []
    for entry in policy.allowed_hosts:
        entry_type = classify_sandbox_network_allowlist_entry(entry)
        if entry_type == SandboxNetworkAllowlistEntryType.IPV4_ADDRESS:
            cidr_blocks.append(f"{entry}/32")
        elif entry_type == SandboxNetworkAllowlistEntryType.IPV6_ADDRESS:
            cidr_blocks.append(f"{entry}/128")
        elif entry_type in {
            SandboxNetworkAllowlistEntryType.IPV4_CIDR_BLOCK,
            SandboxNetworkAllowlistEntryType.IPV6_CIDR_BLOCK,
        }:
            cidr_blocks.append(entry)
        else:
            raise ValueError(
                f"akernel sandbox provider cannot convert {entry_type.value} network allowlist entry: {entry}"
            )
    return cidr_blocks
