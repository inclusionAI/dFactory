"""Sandbox provider adapters."""

from anexus.sandbox.health import SandboxHealthReport, check_sandbox_provider
from anexus.sandbox.network import (
    SandboxNetworkAllowlistEntryType,
    SandboxNetworkMode,
    SandboxNetworkPolicy,
    classify_sandbox_network_allowlist_entry,
    normalize_sandbox_allowed_hosts,
    sandbox_network_policy_from_config,
)
from anexus.sandbox.provider import (
    ArtifactMapping,
    DownloadResult,
    ExecResult,
    FixtureMapping,
    SandboxLaunchSpec,
    SandboxLease,
    SandboxProvider,
    SandboxProviderLimits,
    UploadResult,
)
from anexus.sandbox.registry import SandboxProviderRegistry, SandboxProviderRouter
from anexus.sandbox.resources import (
    SandboxResourceNode,
    SandboxResourceSnapshot,
    read_sandbox_provider_resource_snapshot,
    read_sandbox_provider_resource_snapshots,
)

__all__ = [
    "ArtifactMapping",
    "DownloadResult",
    "ExecResult",
    "FixtureMapping",
    "SandboxHealthReport",
    "SandboxLaunchSpec",
    "SandboxLease",
    "SandboxNetworkAllowlistEntryType",
    "SandboxNetworkMode",
    "SandboxNetworkPolicy",
    "SandboxProvider",
    "SandboxProviderLimits",
    "SandboxProviderRegistry",
    "SandboxProviderRouter",
    "SandboxResourceNode",
    "SandboxResourceSnapshot",
    "UploadResult",
    "classify_sandbox_network_allowlist_entry",
    "check_sandbox_provider",
    "normalize_sandbox_allowed_hosts",
    "read_sandbox_provider_resource_snapshot",
    "read_sandbox_provider_resource_snapshots",
    "sandbox_network_policy_from_config",
]
