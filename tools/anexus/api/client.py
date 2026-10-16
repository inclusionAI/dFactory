"""Small client for the read-only local run API."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


class RunApiError(RuntimeError):
    """Non-success response returned by the local run API."""

    def __init__(
        self,
        *,
        status: int,
        code: str,
        message: str,
        payload: dict[str, Any],
    ) -> None:
        self.status = status
        self.code = code
        self.message = message
        self.payload = payload
        super().__init__(f"{status} {code}: {message}")


class RunApiClient:
    """Synchronous stdlib client for the local read-only run API."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 5.0):
        if not base_url.strip():
            raise ValueError("base_url is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def get(self, path: str) -> dict[str, Any]:
        normalized_path = path if path.startswith("/") else f"/{path}"
        return self._get(normalized_path)

    def inspect(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/runs/{_quote_run_id(run_id)}/inspect")

    def events(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/runs/{_quote_run_id(run_id)}/events")

    def attempts(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/runs/{_quote_run_id(run_id)}/attempts")

    def live(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/runs/{_quote_run_id(run_id)}/live")

    def summary(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/runs/{_quote_run_id(run_id)}/summary")

    def _get(self, path: str) -> dict[str, Any]:
        request = Request(f"{self.base_url}{path}", method="GET")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return _decode_json(response.read())
        except HTTPError as exc:
            payload = _decode_json(exc.read())
            error = payload.get("error", {})
            if not isinstance(error, dict):
                error = {}
            raise RunApiError(
                status=exc.code,
                code=str(error.get("code", "http_error")),
                message=str(error.get("message", exc.reason)),
                payload=payload,
            ) from exc


def _quote_run_id(run_id: str) -> str:
    value = run_id.strip()
    if not value:
        raise ValueError("run_id is required")
    return quote(value, safe="")


def _decode_json(raw: bytes) -> dict[str, Any]:
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("API response must be a JSON object")
    return data
