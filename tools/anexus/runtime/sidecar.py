"""Sidecar readiness checks."""

from __future__ import annotations

import asyncio
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen

from anexus.core.errors import AnexusError, ErrorKind
from anexus.core.execution import SidecarPlan


async def wait_for_sidecar_health(plan: SidecarPlan, base_url: str) -> None:
    """Wait until a sidecar health endpoint succeeds or times out."""

    timeout = float(plan.health_timeout_seconds)
    if timeout <= 0:
        return
    health_url = _health_url(base_url, plan.health_path)
    started = perf_counter()
    deadline = started + timeout
    last_error = ""

    while True:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            break
        try:
            result = await asyncio.to_thread(
                _probe_health_url,
                health_url,
                min(max(remaining, 0.001), 1.0),
            )
            if result is None:
                return
            last_error = result
        except asyncio.CancelledError:
            raise
        await asyncio.sleep(min(0.1, max(deadline - perf_counter(), 0.0)))

    raise AnexusError(
        f"sidecar {plan.name!r} did not become healthy",
        kind=ErrorKind.SANDBOX_ERROR,
        component="sidecar.health",
        details={
            "reason": "sidecar_health_timeout",
            "sidecar_name": plan.name,
            "provider": plan.launch.provider,
            "health_url": health_url,
            "health_timeout_seconds": timeout,
            "elapsed_seconds": perf_counter() - started,
            "last_error": last_error,
        },
    )


def _probe_health_url(url: str, timeout: float) -> str | None:
    try:
        with urlopen(url, timeout=timeout) as response:
            status = int(getattr(response, "status", 200))
            if 200 <= status < 400:
                return None
            return f"HTTP {status}"
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return str(exc) or exc.__class__.__name__


def _health_url(base_url: str, health_path: str) -> str:
    path = health_path.strip()
    if not path:
        return base_url
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
