"""Kubeconfig environment extraction for sidecar sandboxes."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

_FIELD_TO_ENV = {
    "server": "K8S_API_SERVER",
    "certificate-authority-data": "K8S_CA_B64",
    "client-certificate-data": "K8S_CLIENT_CERT_B64",
    "client-key-data": "K8S_CLIENT_KEY_B64",
}


def load_kubeconfig_environment(path: str) -> dict[str, str]:
    """Extract K8S_* values from a local kubeconfig file."""

    source_path = _local_kubeconfig_path(path)
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"sidecar kubeconfig is not readable: {source_path}: {exc}") from exc

    env = {
        env_key: value
        for source_key, env_key in _FIELD_TO_ENV.items()
        if (value := _extract_scalar_value(text, source_key))
    }
    if not env:
        raise ValueError(
            f"sidecar kubeconfig has no supported credentials: {source_path}; "
            f"supported fields: {', '.join(sorted(_FIELD_TO_ENV))}"
        )
    return env


def require_kubeconfig_path(path: str) -> str:
    """Return a non-empty kubeconfig path or raise a structured sandbox error."""

    if path.strip():
        return path
    raise ValueError("sidecar kubeconfig path is required when inject_kubeconfig is enabled")


def _local_kubeconfig_path(path: str) -> Path:
    parts = urlsplit(path)
    if parts.scheme == "file":
        return Path(unquote(parts.path)).expanduser()
    if parts.scheme:
        raise ValueError(f"sidecar kubeconfig must be a local path: {path}")
    return Path(path).expanduser()


def _extract_scalar_value(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}:\s*['\"]?([^'\"\s]+)", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""
