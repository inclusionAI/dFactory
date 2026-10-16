"""Provider-neutral sandbox network policy models."""

from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum  # type: ignore[no-redef,assignment]
from ipaddress import ip_address, ip_network
from typing import Any

_HOST_LABEL_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class SandboxNetworkMode(StrEnum):
    """Network access mode requested for a sandbox."""

    PUBLIC = "public"
    NO_NETWORK = "no-network"
    ALLOWLIST = "allowlist"


class SandboxNetworkAllowlistEntryType(StrEnum):
    """Type of a normalized network allowlist entry."""

    HOSTNAME = "hostname"
    WILDCARD_HOSTNAME = "wildcard-hostname"
    IPV4_ADDRESS = "ipv4-address"
    IPV6_ADDRESS = "ipv6-address"
    IPV4_CIDR_BLOCK = "ipv4-cidr-block"
    IPV6_CIDR_BLOCK = "ipv6-cidr-block"


@dataclass(frozen=True)
class SandboxNetworkPolicy:
    """Resolved sandbox network policy carried by launch specs."""

    network_mode: SandboxNetworkMode | str = SandboxNetworkMode.PUBLIC
    allowed_hosts: tuple[str, ...] | list[str] = ()

    def __post_init__(self) -> None:
        network_mode = coerce_sandbox_network_mode(self.network_mode)
        allowed_hosts = normalize_sandbox_allowed_hosts(self.allowed_hosts)
        if network_mode != SandboxNetworkMode.ALLOWLIST and allowed_hosts:
            raise ValueError("allowed_hosts is only valid when network_mode='allowlist'")
        object.__setattr__(self, "network_mode", network_mode)
        object.__setattr__(self, "allowed_hosts", allowed_hosts)

    @property
    def is_public(self) -> bool:
        return self.network_mode == SandboxNetworkMode.PUBLIC and not self.allowed_hosts

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"network_mode": self.network_mode.value}
        if self.allowed_hosts:
            data["allowed_hosts"] = list(self.allowed_hosts)
        return data


def coerce_sandbox_network_mode(value: SandboxNetworkMode | str) -> SandboxNetworkMode:
    """Coerce a config or API value into a sandbox network mode."""

    if isinstance(value, SandboxNetworkMode):
        return value
    raw_value = str(value).strip().lower()
    try:
        return SandboxNetworkMode(raw_value)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in SandboxNetworkMode)
        raise ValueError(f"invalid sandbox network_mode {value!r}; expected one of: {allowed}") from exc


def sandbox_network_policy_from_config(
    raw: dict[str, Any] | None,
    *,
    default: SandboxNetworkPolicy | None = None,
) -> SandboxNetworkPolicy:
    """Parse network policy fields from a sandbox-like config mapping."""

    if raw is None:
        return default or SandboxNetworkPolicy()
    policy_section = raw.get("network_policy")
    if policy_section is not None and not isinstance(policy_section, dict):
        raise ValueError("network_policy must be a mapping")
    source = policy_section if isinstance(policy_section, dict) else raw
    default_policy = default or SandboxNetworkPolicy()
    network_mode = source.get("network_mode", default_policy.network_mode)
    if "allowed_hosts" in source:
        allowed_hosts = source["allowed_hosts"]
    elif "network_mode" in source:
        explicit_mode = coerce_sandbox_network_mode(network_mode)
        if (
            explicit_mode == SandboxNetworkMode.ALLOWLIST
            and default_policy.network_mode == SandboxNetworkMode.ALLOWLIST
        ):
            allowed_hosts = default_policy.allowed_hosts
        else:
            allowed_hosts = ()
    else:
        allowed_hosts = default_policy.allowed_hosts
    if allowed_hosts is None:
        allowed_hosts = ()
    if isinstance(allowed_hosts, str):
        allowed_hosts = [allowed_hosts]
    if not isinstance(allowed_hosts, list | tuple):
        raise ValueError("allowed_hosts must be a list of host, IP, or CIDR entries")
    return SandboxNetworkPolicy(
        network_mode=network_mode,
        allowed_hosts=tuple(str(entry) for entry in allowed_hosts),
    )


def normalize_sandbox_allowed_hosts(hosts: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate and normalize allowlist entries."""

    normalized: list[str] = []
    for raw_host in hosts:
        host = str(raw_host).strip().lower().rstrip(".")
        if not host:
            raise ValueError("allowed_hosts entries must be non-empty")
        if "://" in host:
            raise ValueError("allowed_hosts entries must be hostnames, IP addresses, or CIDR blocks, not URLs")
        if "/" in host:
            normalized.append(_normalize_cidr_block(host))
            continue
        if ":" in host:
            normalized.append(_normalize_ipv6_address(host))
            continue
        if "[" in host or "]" in host:
            raise ValueError("allowed_hosts entries must not use URL brackets, ports, or paths")
        ip_literal = _normalize_ip_address(host)
        if ip_literal is not None:
            normalized.append(ip_literal)
            continue
        normalized.append(_normalize_hostname_pattern(host))
    return tuple(normalized)


def classify_sandbox_network_allowlist_entry(
    entry: str,
) -> SandboxNetworkAllowlistEntryType:
    """Classify a normalized allowlist entry."""

    if entry.startswith("*."):
        return SandboxNetworkAllowlistEntryType.WILDCARD_HOSTNAME
    if "/" in entry:
        network = ip_network(entry, strict=True)
        if network.version == 4:
            return SandboxNetworkAllowlistEntryType.IPV4_CIDR_BLOCK
        return SandboxNetworkAllowlistEntryType.IPV6_CIDR_BLOCK
    try:
        address = ip_address(entry)
    except ValueError:
        return SandboxNetworkAllowlistEntryType.HOSTNAME
    if address.version == 4:
        return SandboxNetworkAllowlistEntryType.IPV4_ADDRESS
    return SandboxNetworkAllowlistEntryType.IPV6_ADDRESS


def _normalize_ip_address(host: str) -> str | None:
    try:
        address = ip_address(host)
    except ValueError:
        return None
    return address.compressed


def _normalize_ipv6_address(host: str) -> str:
    if "%" in host:
        raise ValueError("allowed_hosts IPv6 zone identifiers are not supported")
    normalized = _normalize_ip_address(host)
    if normalized is None or ip_address(normalized).version != 6:
        raise ValueError("allowed_hosts entries must not include ports or paths")
    return normalized


def _normalize_cidr_block(host: str) -> str:
    try:
        network = ip_network(host, strict=True)
    except ValueError as exc:
        raise ValueError("allowed_hosts CIDR entries must be valid IPv4 or IPv6 CIDR blocks") from exc
    return network.compressed


def _normalize_hostname_pattern(host: str) -> str:
    if host.startswith("*."):
        host_to_validate = host[2:]
        if not host_to_validate:
            raise ValueError("allowed_hosts wildcard entries must include a hostname suffix")
        if _normalize_ip_address(host_to_validate) is not None:
            raise ValueError("allowed_hosts wildcard entries must target hostnames")
    elif "*" in host:
        raise ValueError("allowed_hosts wildcard entries must use a leading '*.' prefix")
    else:
        host_to_validate = host
    labels = host_to_validate.split(".")
    if not labels or not all(_HOST_LABEL_PATTERN.match(label) for label in labels):
        raise ValueError(
            "allowed_hosts entries must be valid hostnames, wildcard hostnames, IP addresses, or CIDR blocks"
        )
    return host
