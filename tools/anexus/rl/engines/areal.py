"""AReaL inference engine connector."""

from __future__ import annotations

from urllib.parse import urlparse


class ArealEngine:
    """Inference engine backed by AReaL's OpenAI-compatible proxy."""

    def __init__(self, openai_base_url: str) -> None:
        parsed = urlparse(openai_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("OPENAI_BASE_URL must be an absolute HTTP URL")

        path_parts = [part for part in parsed.path.strip("/").split("/") if part]
        self._session_id = path_parts[-1] if path_parts else "unknown"
        self._proxy_base = f"{parsed.scheme}://{parsed.netloc}"
        self._path_prefix = "/" + "/".join(path_parts[:-1]) if len(path_parts) >= 2 else ""

    def get_completion_url(self, session_id: str) -> str:
        return f"{self._proxy_base}{self._path_prefix}/{session_id}/chat/completions"

    def get_session_id(self) -> str:
        return self._session_id


class ArealConnector:
    """Connect to AReaL through its trainer-injected ``OPENAI_BASE_URL``."""

    def connect(self, env: dict[str, str]) -> ArealEngine:
        url = env.get("OPENAI_BASE_URL", "")
        if not url:
            raise ValueError("OPENAI_BASE_URL not set")
        return ArealEngine(url)
