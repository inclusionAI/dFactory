"""AKernel resource snapshot normalization."""

from __future__ import annotations

import json
import ssl
from collections.abc import Iterable, Mapping
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from anexus.config.catalog import ControlPlaneSpec
from anexus.core.errors import AnexusError, ErrorKind
from anexus.sandbox.resources import SandboxResourceNode, SandboxResourceSnapshot


def query_akernel_resources(control_plane: ControlPlaneSpec) -> dict[str, Any]:
    """Query the same resource endpoint used by ``ak resources``."""

    server = control_plane.server_address
    token = control_plane.token

    url = f"https://{server}/global-scheduler/resources"
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    if not control_plane.tls.verify:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    req = request.Request(url, method="GET", headers={"X-Auth": token, "Type": "json"})
    cause: BaseException | None = None
    try:
        with request.urlopen(req, context=context, timeout=30) as response:
            body = response.read().decode("utf-8")
            status = int(response.status)
    except HTTPError as exc:
        cause = exc
        body = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
        message = f"akernel resource query failed with HTTP {status}"
        retryable = 500 <= status < 600
        details: dict[str, object] = {
            "url": url,
            "status": status,
            "response": body[:2000],
        }
    except URLError as exc:
        cause = exc
        message = f"akernel resource query request failed: {exc.reason}"
        retryable = True
        details = {
            "url": url,
            "exception_type": exc.__class__.__name__,
            "reason": str(exc.reason),
        }
    else:
        if status != 200:
            message = f"akernel resource query failed with HTTP {status}"
            retryable = 500 <= status < 600
            details = {"url": url, "status": status, "response": body[:2000]}
        else:
            try:
                return json.loads(body)
            except json.JSONDecodeError as exc:
                cause = exc
                message = f"akernel resource query returned invalid JSON: {exc.msg}"
                retryable = True
                details = {"url": url, "status": status, "response": body[:2000]}

    error = AnexusError(
        message,
        kind=ErrorKind.INFRASTRUCTURE_ERROR,
        component="sandbox.akernel",
        retryable=retryable,
        details={"operation": "resource_snapshot", **details},
    )
    if cause is not None:
        raise error from cause
    raise error


def normalize_akernel_resource_snapshot(
    raw_resources: object,
    *,
    provider_name: str = "akernel",
) -> SandboxResourceSnapshot:
    """Normalize AKernel SDK resource list and query-resource fragment shapes."""

    nodes: list[SandboxResourceNode] = []
    for unit in _resource_units(raw_resources):
        if not isinstance(unit, Mapping):
            continue
        capacity = _extract_resource_map(unit.get("capacity", {}))
        allocatable = _extract_resource_map(unit.get("allocatable", {}))
        nodes.append(
            SandboxResourceNode(
                node_id=str(unit.get("id") or unit.get("name") or ""),
                status=int(_extract_scalar(unit.get("status", 0))),
                cpu_capacity_m=capacity.get("CPU", 0.0),
                cpu_allocatable_m=allocatable.get("CPU", 0.0),
                memory_capacity_mb=capacity.get("Memory", 0.0),
                memory_allocatable_mb=allocatable.get("Memory", 0.0),
                labels=_extract_labels(unit.get("labels") or unit.get("nodeLabels") or {}),
            )
        )
    return SandboxResourceSnapshot(provider_name=provider_name, nodes=tuple(nodes))


def _resource_units(raw_resources: object) -> Iterable[object]:
    if isinstance(raw_resources, Mapping):
        resource = raw_resources.get("resource", raw_resources)
        if isinstance(resource, Mapping):
            fragment = resource.get("fragment")
            if isinstance(fragment, Mapping):
                return list(fragment.values())
            return [resource]
        if isinstance(resource, Iterable) and not isinstance(resource, str | bytes):
            return resource
        return []
    if isinstance(raw_resources, Iterable) and not isinstance(raw_resources, str | bytes):
        return raw_resources
    return []


def _extract_resource_map(raw: object) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    inner = raw.get("resources", raw)
    if not isinstance(inner, Mapping):
        return {}
    result: dict[str, float] = {}
    for name, value in inner.items():
        result[str(name)] = _extract_scalar(value)
    return result


def _extract_scalar(value: object) -> float:
    raw = value
    if isinstance(raw, Mapping) and "scalar" in raw:
        raw = raw.get("scalar")
    if isinstance(raw, Mapping) and "value" in raw:
        raw = raw.get("value")
    try:
        return float(raw or 0)
    except (TypeError, ValueError):
        return 0.0


def _extract_labels(raw: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, Mapping):
        return {}
    labels: dict[str, tuple[str, ...]] = {}
    for key, value in raw.items():
        if isinstance(value, Mapping) and isinstance(value.get("items"), Mapping):
            labels[str(key)] = tuple(str(item) for item in value["items"])
        elif isinstance(value, list | tuple | set):
            labels[str(key)] = tuple(str(item) for item in value)
        elif value is None:
            labels[str(key)] = ()
        else:
            labels[str(key)] = (str(value),)
    return labels
