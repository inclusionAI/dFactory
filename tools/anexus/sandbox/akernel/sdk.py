"""Async AKernel SDK control-plane helpers."""

from __future__ import annotations

import asyncio
import inspect
import signal
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from akernel_sdk import Sandbox

from anexus.core.errors import AnexusError, ErrorKind

SandboxFactory = Callable[..., Any]


class AkernelSdkControlPlane:
    """Async adapter for the public AKernel SDK.

    A single worker serializes calls for one sandbox. Control-plane operations
    that address independent sandbox names may opt into bounded concurrency.
    SDK object traversal and synchronous calls stay behind this boundary so the
    provider can remain lease-oriented and async.
    """

    def __init__(
        self,
        *,
        max_workers: int = 1,
        sandbox_factory: SandboxFactory | None = None,
    ) -> None:
        """Create an AKernel SDK control plane."""

        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._sandbox_factory = Sandbox if sandbox_factory is None else sandbox_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="akernel-sdk",
        )
        self._closed = False
        self._pending_calls = 0
        self._state_lock = threading.Lock()

    @property
    def busy(self) -> bool:
        """Return whether native SDK work is running or queued."""

        with self._state_lock:
            return self._pending_calls > 0

    def new_lane(self, *, max_workers: int = 1) -> AkernelSdkControlPlane:
        """Create an independent worker lane using the same SDK factory."""

        return AkernelSdkControlPlane(
            max_workers=max_workers,
            sandbox_factory=self._sandbox_factory,
        )

    async def create_sandbox(self, kwargs: Mapping[str, Any]) -> Any:
        """Create a sandbox through the configured public SDK factory."""

        return await self.call(self._sandbox_factory, **dict(kwargs))

    @staticmethod
    def sandbox_id(sandbox: Any) -> str:
        """Return the public SDK's physical sandbox identifier."""

        try:
            sandbox_id = sandbox.id
        except AttributeError as exc:
            raise AnexusError(
                "akernel_sdk.Sandbox did not expose id",
                kind=ErrorKind.SANDBOX_ERROR,
                component="sandbox.akernel",
                retryable=True,
                details={"expected_attribute": "id"},
            ) from exc
        if sandbox_id:
            return str(sandbox_id)
        raise AnexusError(
            "akernel_sdk.Sandbox returned an empty id",
            kind=ErrorKind.SANDBOX_ERROR,
            component="sandbox.akernel",
            retryable=True,
            details={"expected_attribute": "id"},
        )

    async def run_command(
        self,
        sandbox: Any,
        command: str,
        *,
        background: bool = False,
        stdin: bool = False,
        timeout: int | None = None,
    ) -> Any:
        """Run a command through ``Sandbox.commands``."""

        kwargs: dict[str, Any] = {}
        if background:
            kwargs["background"] = True
        if stdin:
            kwargs["stdin"] = True
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await self.call(sandbox.commands.run, command, **kwargs)

    async def send_stdin(
        self,
        handle: Any,
        data: str,
        *,
        eof: bool = False,
    ) -> Any:
        """Send input to a background command handle."""

        return await self.call(handle.send_stdin, data, eof=eof)

    async def wait_command(self, handle: Any, *, timeout: int | None = None) -> Any:
        """Wait for a background command handle."""

        return await self.call(handle.wait, timeout=timeout)

    async def copy_from_local(
        self,
        sandbox: Any,
        source_path: str,
        target_path: str,
    ) -> None:
        """Copy a local path into a sandbox through the SDK filesystem API."""

        await self.call(sandbox.files.copy_from_local, source_path, target_path)

    async def copy_to_local(
        self,
        sandbox: Any,
        source_path: str,
        target_path: str,
    ) -> None:
        """Copy a sandbox path to the local filesystem through the SDK."""

        await self.call(sandbox.files.copy_to_local, source_path, target_path)

    async def get_port_url(
        self,
        sandbox: Any,
        port: int,
        *,
        internal: bool = False,
    ) -> str:
        """Resolve a declared sandbox port through the SDK."""

        return str(await self.call(sandbox.get_port_url, port, internal=internal))

    async def kill_sandbox(self, sandbox: Any) -> None:
        """Terminate or close one SDK sandbox handle."""

        await self.call(sandbox.kill)

    async def delete_sandbox(self, name: str) -> None:
        """Delete a named sandbox through the public SDK lifecycle API."""

        await self.call(self._sandbox_factory.delete, name)

    async def call(self, call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run one SDK call on the serialized worker.

        Args:
            call: AKernel SDK callable.
            *args: Positional arguments passed to `call`.
            **kwargs: Keyword arguments passed to `call`.

        Returns:
            Result returned by the SDK callable. Awaitable SDK results are
            driven to completion inside the worker.

        Raises:
            RuntimeError: If the control plane has been closed.
        """

        with self._state_lock:
            if self._closed:
                raise RuntimeError("AKernel SDK control plane is closed")
            self._pending_calls += 1
        loop = asyncio.get_running_loop()
        sigint_handler = (
            signal.getsignal(signal.SIGINT) if threading.current_thread() is threading.main_thread() else None
        )

        def run_sdk_call() -> Any:
            result = call(*args, **kwargs)
            if not inspect.isawaitable(result):
                return result
            return asyncio.run(result)

        try:
            future = self._executor.submit(run_sdk_call)
        except BaseException:
            with self._state_lock:
                self._pending_calls -= 1
            raise
        future.add_done_callback(self._call_finished)
        if sigint_handler is not None:
            future.add_done_callback(lambda _future: _schedule_sigint_restore(loop, sigint_handler))
        try:
            return await asyncio.wrap_future(future, loop=loop)
        finally:
            _restore_sigint_handler(sigint_handler)

    def close(self, *, wait: bool = True) -> None:
        """Release the worker thread."""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _call_finished(self, _future: object) -> None:
        """Mark one submitted native SDK call complete."""

        with self._state_lock:
            self._pending_calls -= 1


def _schedule_sigint_restore(
    loop: asyncio.AbstractEventLoop,
    handler: Any,
) -> None:
    """Restore SIGINT on the event-loop thread after native SDK work."""

    try:
        loop.call_soon_threadsafe(_restore_sigint_handler, handler)
    except RuntimeError:
        return


def _restore_sigint_handler(handler: Any) -> None:
    """Reapply a Python SIGINT handler that native SDK code may replace."""

    if handler is None or threading.current_thread() is not threading.main_thread():
        return
    if signal.getsignal(signal.SIGINT) == handler:
        signal.signal(signal.SIGINT, handler)
