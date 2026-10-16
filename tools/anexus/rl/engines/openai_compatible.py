"""Shared OpenAI-compatible inference engine helpers."""

from __future__ import annotations

from urllib.parse import urlparse


class OpenAICompatibleEngine:
    """Inference engine backed by an OpenAI-compatible HTTP server."""

    def __init__(self, base_url: str, *, session_id: str = "default") -> None:
        self._base_url = normalize_openai_base_url(base_url, label="base_url")
        self._session_id = session_id or "default"

    def get_completion_url(self, session_id: str) -> str:
        return f"{self._base_url}/chat/completions"

    def get_session_id(self) -> str:
        return self._session_id


def normalize_openai_base_url(url: str, *, label: str) -> str:
    """Return an absolute HTTP base URL ending at the OpenAI-compatible v1 root."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an absolute HTTP URL")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def base_url_from_env(
    env: dict[str, str],
    *,
    url_keys: tuple[str, ...],
    host_key: str,
    port_key: str,
    label: str,
) -> str:
    """Resolve an OpenAI-compatible base URL from URL or host/port env vars."""

    for key in url_keys:
        value = env.get(key, "")
        if value:
            return normalize_openai_base_url(value, label=key)

    host = env.get(host_key, "")
    port = env.get(port_key, "")
    if host and port:
        return normalize_openai_base_url(f"http://{host}:{port}", label=f"{host_key}/{port_key}")
    raise ValueError(f"{label} base URL not set")


def session_id_from_env(
    env: dict[str, str],
    *,
    keys: tuple[str, ...],
    default: str = "default",
) -> str:
    """Return the first non-empty session id from env."""

    for key in keys:
        value = env.get(key, "")
        if value:
            return value
    return default
