"""AKernel SDK sandbox provider adapter.

Runtime concurrency is admitted by the queue. Global AKernel SDK operations are
serialized, while each sandbox has its own serialized SDK lane so commands in
different sandboxes can run concurrently. The provider exposes an async
lease-oriented interface and keeps provider-native SDK handles private.
"""

from __future__ import annotations

import asyncio
import logging
import os
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from anexus.config.catalog import ObjectStoreSpec, SandboxProviderSpec
from anexus.core.context import TaskContext
from anexus.core.errors import AnexusError, ErrorKind
from anexus.sandbox.akernel.errors import (
    classify_akernel_error,
)
from anexus.sandbox.akernel.errors import (
    delete_failure_means_remote_gone as _delete_failure_means_remote_gone,
)
from anexus.sandbox.akernel.launch import (
    build_sandbox_kwargs,
)
from anexus.sandbox.akernel.launch import (
    file_override_paths as _file_override_paths,
)
from anexus.sandbox.akernel.oss import (
    copy_between_temp_uri as _copy_between_temp_uri,
)
from anexus.sandbox.akernel.oss import (
    exec_failure_summary as _exec_failure_summary,
)
from anexus.sandbox.akernel.oss import (
    oss_download_command as _oss_download_command,
)
from anexus.sandbox.akernel.oss import (
    oss_upload_command as _oss_upload_command,
)
from anexus.sandbox.akernel.resources import (
    normalize_akernel_resource_snapshot,
)
from anexus.sandbox.akernel.resources import (
    query_akernel_resources as _query_akernel_resources_http,
)
from anexus.sandbox.akernel.sdk import AkernelSdkControlPlane, SandboxFactory
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
from anexus.sandbox.resources import SandboxResourceSnapshot

logger = logging.getLogger(__name__)
_CLEANUP_MAX_WORKERS = 8


class AkernelSandboxProvider(SandboxProvider):
    """`SandboxProvider` implementation backed by `akernel_sdk.Sandbox`.

    Global SDK operations and calls for each individual sandbox are serialized.
    Separate sandboxes use separate SDK lanes so long-running commands preserve
    runtime concurrency without overlapping SDK calls on the same sandbox.

    Attributes:
        provider: Provider name used in `SandboxLease.provider` and router
            dispatch.
    """

    def __init__(
        self,
        *,
        spec: SandboxProviderSpec,
        object_stores: dict[str, ObjectStoreSpec],
        sandbox_factory: SandboxFactory | None = None,
        active_sandbox_limit: int | None = None,
        sdk_control_plane: AkernelSdkControlPlane | None = None,
    ):
        """Initialize the AKernel provider."""
        if spec.driver != "akernel":
            raise ValueError(f"unsupported AKernel provider driver: {spec.driver}")
        if spec.data_transfer.driver != "aliyun-oss":
            raise ValueError(f"unsupported AKernel data-transfer driver: {spec.data_transfer.driver}")
        self.provider = spec.name
        self._spec = spec
        self._object_stores = dict(object_stores)
        self._sandboxes: dict[str, Any] = {}
        self._sandbox_sdks: dict[str, AkernelSdkControlPlane] = {}
        self._sandbox_names: dict[str, str] = {}
        self._detached_sandbox_ids: set[str] = set()
        self._confirmed_gone_sandbox_ids: set[str] = set()
        self._active_sandbox_limit = active_sandbox_limit
        self._sdk = sdk_control_plane or AkernelSdkControlPlane(sandbox_factory=sandbox_factory)
        self._owns_sdk_control_plane = sdk_control_plane is None
        self._cleanup_sdk = self._sdk.new_lane(max_workers=_CLEANUP_MAX_WORKERS)
        self._saved_sdk_environment = self._activate_sdk_environment()

    def limits(self) -> SandboxProviderLimits:
        """Return scheduler-visible capacity for this provider."""

        return SandboxProviderLimits(
            provider_name=self.provider,
            active_sandboxes=self._active_sandbox_limit,
        )

    def control_plane_env(self, _provider_name: str = "") -> dict[str, str]:
        """Return AKernel control-plane env for nested sandbox users."""

        control_plane = self._spec.control_plane
        return {
            "AKERNEL_SERVER_ADDRESS": (control_plane.internal_address or control_plane.server_address),
            "AKERNEL_GATEWAY_ADDRESS": control_plane.gateway_address,
            "AKERNEL_TOKEN": control_plane.token,
        }

    def close(self) -> None:
        """Release provider-owned AKernel SDK resources."""

        for sdk in self._sandbox_sdks.values():
            sdk.close(wait=False)
        self._sandbox_sdks.clear()
        self._cleanup_sdk.close(wait=False)
        if self._owns_sdk_control_plane:
            self._sdk.close(wait=False)
        self._restore_sdk_environment()

    async def create(
        self,
        context: TaskContext,
        spec: SandboxLaunchSpec | None = None,
    ) -> SandboxLease:
        """Allocate an AKernel sandbox and run the launch bootstrap command."""
        if spec is None:
            raise ValueError("akernel sandbox launch spec is required")
        provider_name = spec.provider or self.provider

        sandbox_kwargs = self._sandbox_kwargs(context, spec)
        sandbox_name = str(sandbox_kwargs["name"])
        detached = bool(sandbox_kwargs.get("detached", False))
        create_task = asyncio.create_task(self._sdk.create_sandbox(sandbox_kwargs))
        try:
            sandbox = await asyncio.shield(create_task)
        except asyncio.CancelledError as cancelled:
            try:
                sandbox = await create_task
            except BaseException as create_exc:
                cancelled.add_note(f"AKernel sandbox creation did not complete during cancellation: {create_exc}")
                raise cancelled from None
            try:
                await self._delete_native_sandbox(
                    sandbox,
                    sandbox_name=sandbox_name,
                    detached=detached,
                    sdk=self._sdk,
                )
            except BaseException as cleanup_exc:
                cancelled.add_note(f"AKernel sandbox cleanup after cancelled creation failed: {cleanup_exc}")
            raise cancelled from None
        except AnexusError:
            raise
        except Exception as exc:
            raise classify_akernel_error(exc, operation="create") from exc
        try:
            sandbox_id = self._sdk.sandbox_id(sandbox)
            await self._apply_file_overrides_async(context, sandbox, spec)
            if spec.command:
                try:
                    await self._sdk.run_command(sandbox, spec.command, background=True)
                except AnexusError:
                    raise
                except Exception as exc:
                    base = classify_akernel_error(
                        exc,
                        operation="start_command",
                    )
                    error = AnexusError(
                        f"akernel startup command failed for sandbox {sandbox_id}",
                        kind=base.kind,
                        component=base.component,
                        retryable=base.retryable,
                        details={
                            **base.details,
                            "sandbox_id": sandbox_id,
                            "command": spec.command,
                        },
                    )
                    raise error from exc
        except BaseException as exc:
            try:
                await self._delete_native_sandbox(
                    sandbox,
                    sandbox_name=sandbox_name,
                    detached=detached,
                    sdk=self._sdk,
                )
            except BaseException as cleanup_exc:
                exc.add_note(f"AKernel sandbox cleanup after create failure also failed: {cleanup_exc}")
            raise
        self._sandboxes[sandbox_id] = sandbox
        self._sandbox_sdks[sandbox_id] = self._sdk.new_lane()
        self._sandbox_names[sandbox_id] = sandbox_name
        if detached:
            self._detached_sandbox_ids.add(sandbox_id)
        return SandboxLease(sandbox_id=sandbox_id, provider=provider_name)

    async def exec(
        self,
        lease: SandboxLease,
        command: str,
        *,
        stdin: str = "",
        timeout: int | float | None = None,
    ) -> ExecResult:
        """Run a shell command inside the leased sandbox."""

        sandbox = self._require_sandbox(lease)
        sdk = self._sdk_for_sandbox(lease.sandbox_id)
        try:
            if stdin:
                handle = await sdk.run_command(
                    sandbox,
                    command,
                    background=True,
                    stdin=True,
                )
                await sdk.send_stdin(handle, stdin, eof=True)
                result = await sdk.wait_command(
                    handle,
                    timeout=int(timeout) if timeout is not None else None,
                )
            elif timeout is None:
                result = await sdk.run_command(sandbox, command)
            else:
                result = await sdk.run_command(
                    sandbox,
                    command,
                    timeout=int(timeout),
                )
            return ExecResult(
                exit_code=int(result.exit_code),
                stdout=str(result.stdout),
                stderr=str(result.stderr),
            )
        except AnexusError:
            raise
        except Exception as exc:
            base = classify_akernel_error(exc, operation="exec")
            error = AnexusError(
                str(base),
                kind=base.kind,
                component=base.component,
                retryable=base.retryable,
                details={
                    **base.details,
                    "provider": lease.provider,
                    "sandbox_id": lease.sandbox_id,
                },
            )
            raise error from exc

    async def put_file(self, lease: SandboxLease, source_path: str | Path, target_path: str) -> None:
        """Copy a local file or directory into a sandbox."""

        sandbox = self._require_sandbox(lease)
        sdk = self._sdk_for_sandbox(lease.sandbox_id)
        try:
            await sdk.copy_from_local(sandbox, str(source_path), target_path)
        except AnexusError:
            raise
        except Exception as exc:
            raise classify_akernel_error(exc, operation="put_file") from exc

    async def get_file(self, lease: SandboxLease, source_path: str, target_path: str | Path) -> None:
        """Copy a file or directory from a sandbox to the local filesystem."""

        sandbox = self._require_sandbox(lease)
        sdk = self._sdk_for_sandbox(lease.sandbox_id)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            await sdk.copy_to_local(sandbox, source_path, str(target))
        except AnexusError:
            raise
        except Exception as exc:
            raise classify_akernel_error(exc, operation="get_file") from exc

    async def port_url(self, lease: SandboxLease, port: int, *, internal: bool = False) -> str:
        """Resolve an AKernel gateway URL for a forwarded sandbox port."""

        sandbox = self._require_sandbox(lease)
        sdk = self._sdk_for_sandbox(lease.sandbox_id)
        try:
            return await sdk.get_port_url(sandbox, port, internal=internal)
        except AnexusError:
            raise
        except Exception as exc:
            raise classify_akernel_error(exc, operation="port_url") from exc

    async def delete(self, lease: SandboxLease) -> None:
        """Terminate the leased AKernel sandbox."""

        sandbox = self._sandboxes.get(lease.sandbox_id)
        confirmed_gone = lease.sandbox_id in self._confirmed_gone_sandbox_ids
        if sandbox is None:
            if confirmed_gone:
                return
            raise AnexusError(
                f"unknown akernel sandbox lease: {lease.sandbox_id}",
                kind=ErrorKind.SANDBOX_ERROR,
                component="sandbox.akernel",
                retryable=True,
                details={
                    "operation": "delete",
                    "reason": "unknown_local_lease",
                    "provider": lease.provider,
                    "sandbox_id": lease.sandbox_id,
                },
            )
        remote_gone = False
        sdk = self._sdk_for_sandbox(lease.sandbox_id)
        sandbox_name = self._sandbox_names.get(lease.sandbox_id, "")
        detached = lease.sandbox_id in self._detached_sandbox_ids
        try:
            await self._delete_native_sandbox(
                sandbox,
                sandbox_name=sandbox_name,
                detached=detached,
                sdk=sdk,
            )
        except AnexusError as exc:
            remote_gone = _delete_failure_means_remote_gone(exc)
            if not remote_gone:
                raise self._sandbox_delete_error(
                    exc,
                    lease=lease,
                    sandbox_name=sandbox_name,
                    detached=detached,
                ) from exc
        except Exception as exc:
            error = classify_akernel_error(exc, operation="delete")
            remote_gone = _delete_failure_means_remote_gone(error)
            if not remote_gone:
                raise self._sandbox_delete_error(
                    error,
                    lease=lease,
                    sandbox_name=sandbox_name,
                    detached=detached,
                ) from exc
        self._sandboxes.pop(lease.sandbox_id, None)
        self._close_sandbox_sdk(lease.sandbox_id)
        self._sandbox_names.pop(lease.sandbox_id, None)
        self._detached_sandbox_ids.discard(lease.sandbox_id)
        if remote_gone:
            self._confirmed_gone_sandbox_ids.add(lease.sandbox_id)
        else:
            self._confirmed_gone_sandbox_ids.discard(lease.sandbox_id)

    async def resource_snapshot(self) -> SandboxResourceSnapshot:
        """Read AKernel resource availability."""

        try:
            snapshot = await asyncio.to_thread(
                _query_akernel_resources_http,
                self._spec.control_plane,
            )
        except AnexusError:
            raise
        except Exception as exc:
            raise classify_akernel_error(exc, operation="resource_snapshot") from exc
        return normalize_akernel_resource_snapshot(snapshot, provider_name=self.provider)

    # ──── Cluster-internal data operations ──────────────────────────

    async def download_fixtures(
        self,
        lease: SandboxLease,
        fixtures: list[FixtureMapping],
    ) -> list[DownloadResult]:
        """Download OSS fixtures into a sandbox via in-sandbox `ossutil`."""
        results: list[DownloadResult] = []
        for f in fixtures:
            try:
                if not f.source_uri.strip():
                    raise ValueError("fixture source_uri is required")
                if not f.target_path.strip():
                    raise ValueError("fixture target_path is required")
                object_store = self._object_store_for_uri(f.source_uri)
                cmd = _oss_download_command(
                    f,
                    cli_path=self._spec.data_transfer.cli_path,
                    object_store=object_store,
                )
            except ValueError as exc:
                raise AnexusError(
                    str(exc),
                    kind=ErrorKind.ARTIFACT_ERROR,
                    component="sandbox.download_fixtures",
                    details={
                        "source_uri": f.source_uri,
                        "target_path": f.target_path,
                        "kind": f.kind,
                    },
                ) from exc
            result = await self.exec(lease, cmd, timeout=600)
            if result.exit_code != 0:
                raise AnexusError(
                    f"fixture download failed: {_exec_failure_summary(result)}",
                    kind=ErrorKind.ARTIFACT_ERROR,
                    component="sandbox.download_fixtures",
                    retryable=True,
                    details={
                        "source_uri": f.source_uri,
                        "target_path": f.target_path,
                        "kind": f.kind,
                        "exit_code": result.exit_code,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    },
                )
            results.append(
                DownloadResult(
                    source_uri=f.source_uri,
                    target_path=f.target_path,
                    kind=f.kind,
                )
            )
        return results

    async def upload_artifacts(
        self,
        lease: SandboxLease,
        artifacts: list[ArtifactMapping],
    ) -> list[UploadResult]:
        """Upload sandbox artifacts to OSS via in-sandbox `ossutil`."""
        results: list[UploadResult] = []
        for a in artifacts:
            try:
                if not a.source_path.strip():
                    raise ValueError("artifact source_path is required")
                if not a.target_uri.strip():
                    raise ValueError("artifact target_uri is required")
                object_store = self._object_store_for_uri(a.target_uri)
                cmd = _oss_upload_command(
                    a,
                    cli_path=self._spec.data_transfer.cli_path,
                    object_store=object_store,
                )
            except ValueError as exc:
                raise AnexusError(
                    str(exc),
                    kind=ErrorKind.ARTIFACT_ERROR,
                    component="sandbox.upload_artifacts",
                    details={
                        "source_path": a.source_path,
                        "target_uri": a.target_uri,
                        "kind": a.kind,
                    },
                ) from exc

            result = await self.exec(lease, cmd, timeout=600)
            if result.exit_code != 0:
                raise AnexusError(
                    f"artifact upload failed: {_exec_failure_summary(result)}",
                    kind=ErrorKind.ARTIFACT_ERROR,
                    component="sandbox.upload_artifacts",
                    retryable=True,
                    details={
                        "source_path": a.source_path,
                        "target_uri": a.target_uri,
                        "kind": a.kind,
                        "exit_code": result.exit_code,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    },
                )
            results.append(
                UploadResult(
                    source_path=a.source_path,
                    target_uri=a.target_uri,
                    kind=a.kind,
                )
            )
        return results

    async def copy_between(
        self,
        source_lease: SandboxLease,
        source_path: str,
        target_lease: SandboxLease,
        target_path: str,
        *,
        kind: str = "file",
    ) -> None:
        """Copy data between AKernel sandboxes through temporary OSS storage."""
        path_hash = sha256(f"{source_lease.sandbox_id}:{source_path}".encode()).hexdigest()[:16]
        try:
            object_store = self._copy_between_object_store()
            tmp_uri = _copy_between_temp_uri(
                source_lease,
                path_hash,
                object_store=object_store,
            )
        except ValueError as exc:
            raise AnexusError(
                str(exc),
                kind=ErrorKind.ARTIFACT_ERROR,
                component="sandbox.copy_between",
                details={
                    "source_sandbox_id": source_lease.sandbox_id,
                    "target_sandbox_id": target_lease.sandbox_id,
                    "source_path": source_path,
                    "target_path": target_path,
                    "kind": kind,
                },
            ) from exc
        await self.upload_artifacts(
            source_lease,
            [ArtifactMapping(source_path, tmp_uri, kind)],
        )
        await self.download_fixtures(
            target_lease,
            [FixtureMapping(tmp_uri, target_path, kind)],
        )

    async def _delete_native_sandbox(
        self,
        sandbox: Any,
        *,
        sandbox_name: str,
        detached: bool,
        sdk: AkernelSdkControlPlane,
    ) -> None:
        """Terminate remotely without waiting behind in-sandbox SDK work."""

        if not sandbox_name:
            raise RuntimeError("AKernel sandbox is missing its logical name")

        local_state_closed = False
        if detached and not sdk.busy:
            try:
                await sdk.kill_sandbox(sandbox)
            except Exception:
                logger.debug(
                    "failed to close local state for AKernel sandbox %s before deletion",
                    sandbox_name,
                    exc_info=True,
                )
            else:
                local_state_closed = True

        await self._cleanup_sdk.delete_sandbox(sandbox_name)

        # A busy sandbox lane may still contain a cancelled commands.run call.
        # Never queue local cleanup behind it; remote deletion remains the
        # cancellation-safe path.
        if not local_state_closed and not sdk.busy:
            try:
                await sdk.kill_sandbox(sandbox)
            except Exception:
                logger.debug(
                    "failed to close local state for deleted AKernel sandbox %s",
                    sandbox_name,
                    exc_info=True,
                )

    @staticmethod
    def _sandbox_delete_error(
        error: AnexusError,
        *,
        lease: SandboxLease,
        sandbox_name: str,
        detached: bool,
    ) -> AnexusError:
        """Attach stable resource identity to a failed sandbox deletion."""

        return AnexusError(
            str(error),
            kind=error.kind,
            component=error.component,
            retryable=error.retryable,
            details={
                **error.details,
                "provider": lease.provider,
                "sandbox_id": lease.sandbox_id,
                "akernel_sandbox_name": sandbox_name,
                "detached": detached,
            },
        )

    def _sandbox_kwargs(self, context: TaskContext, spec: SandboxLaunchSpec) -> dict[str, Any]:
        """Translate a provider launch spec into `akernel_sdk.Sandbox` kwargs."""

        return build_sandbox_kwargs(context, spec)

    async def _apply_file_overrides_async(
        self,
        context: TaskContext,
        sandbox: Any,
        spec: SandboxLaunchSpec,
    ) -> None:
        """Copy launch-time file overrides into a sandbox."""

        for index, override in enumerate(spec.file_overrides):
            try:
                source_path, target_path = _file_override_paths(override, index=index)
            except (OSError, ValueError) as exc:
                raise AnexusError(
                    str(exc),
                    kind=ErrorKind.SANDBOX_ERROR,
                    component="sandbox.akernel",
                    details={
                        "operation": "file_override",
                        "reason": "file_override_validation_failed",
                        "index": index,
                        "source_path": str(override.get("src", "")),
                        "target_path": str(override.get("dst", "")),
                        "exception_type": exc.__class__.__name__,
                        **context.to_fields(),
                    },
                ) from exc
            try:
                await self._sdk.copy_from_local(
                    sandbox,
                    str(source_path),
                    target_path,
                )
            except AnexusError:
                raise
            except Exception as exc:
                base = classify_akernel_error(
                    exc,
                    operation="file_override",
                )
                error = AnexusError(
                    f"akernel file override failed while copying {source_path} to {target_path}",
                    kind=base.kind,
                    component=base.component,
                    retryable=base.retryable,
                    details={
                        **base.details,
                        "index": index,
                        "source_path": str(source_path),
                        "target_path": target_path,
                    },
                )
                raise error from exc

    def _object_store_for_uri(self, uri: str) -> ObjectStoreSpec:
        parts = urlsplit(uri)
        if parts.scheme != "oss" or not parts.netloc:
            raise ValueError(f"OSS URI must use oss://bucket/path: {uri}")
        matches = [store for store in self._object_stores.values() if store.bucket == parts.netloc]
        if len(matches) != 1:
            raise ValueError(f"OSS bucket {parts.netloc!r} must match exactly one configured object store")
        object_store = matches[0]
        if object_store.driver != self._spec.data_transfer.driver:
            raise ValueError(
                f"object store driver {object_store.driver!r} does not match "
                f"data-transfer driver {self._spec.data_transfer.driver!r}"
            )
        return object_store

    def _copy_between_object_store(self) -> ObjectStoreSpec:
        if len(self._object_stores) != 1:
            raise ValueError("cross-sandbox copy requires exactly one configured object store")
        object_store = next(iter(self._object_stores.values()))
        if object_store.driver != self._spec.data_transfer.driver:
            raise ValueError(
                f"object store driver {object_store.driver!r} does not match "
                f"data-transfer driver {self._spec.data_transfer.driver!r}"
            )
        return object_store

    def _activate_sdk_environment(self) -> dict[str, str | None]:
        control_plane = self._spec.control_plane
        values = {
            "AKERNEL_SERVER_ADDRESS": control_plane.server_address,
            "AKERNEL_GATEWAY_ADDRESS": control_plane.gateway_address,
            "AKERNEL_TOKEN": control_plane.token,
        }
        previous = {key: os.environ.get(key) for key in values}
        os.environ.update(values)
        return previous

    def _restore_sdk_environment(self) -> None:
        previous = self._saved_sdk_environment
        self._saved_sdk_environment = {}
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _require_sandbox(self, lease: SandboxLease) -> Any:
        """Return the provider-native sandbox handle for a lease."""

        sandbox = self._sandboxes.get(lease.sandbox_id)
        if sandbox is not None:
            return sandbox
        raise AnexusError(
            f"unknown akernel sandbox lease: {lease.sandbox_id}",
            kind=ErrorKind.SANDBOX_ERROR,
            component="sandbox.akernel",
            details={"sandbox_id": lease.sandbox_id},
        )

    def _sdk_for_sandbox(self, sandbox_id: str) -> AkernelSdkControlPlane:
        """Return the serialized SDK lane for one sandbox."""

        return self._sandbox_sdks.get(sandbox_id, self._sdk)

    def _close_sandbox_sdk(self, sandbox_id: str) -> None:
        """Release a sandbox SDK lane after its remote resource is gone."""

        sdk = self._sandbox_sdks.pop(sandbox_id, None)
        if sdk is not None:
            sdk.close(wait=False)
