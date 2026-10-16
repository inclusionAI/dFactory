"""Read-only HTTP API for replaying local run state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import unquote, urlsplit

from anexus.api.inspection import (
    build_run_attempts_payload,
    build_run_events_payload,
    build_run_inspection,
    build_run_live_payload,
    build_run_summary_payload,
)
from anexus.runtime import LocalRunStore


@dataclass
class RunApiServer:
    """Handle for a background read-only run API server."""

    httpd: ThreadingHTTPServer
    thread: Thread
    url: str

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def __enter__(self) -> RunApiServer:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def serve_run_api(
    output_root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> RunApiServer:
    """Start a local read-only HTTP API over a :class:`LocalRunStore`."""

    store = LocalRunStore(output_root)
    handler = _make_handler(store)
    httpd = ThreadingHTTPServer((host, port), handler)
    actual_host, actual_port = httpd.server_address[:2]
    thread = Thread(target=httpd.serve_forever, name="anexus-run-api", daemon=True)
    thread.start()
    return RunApiServer(
        httpd=httpd,
        thread=thread,
        url=f"http://{actual_host}:{actual_port}",
    )


def _make_handler(store: LocalRunStore) -> type[BaseHTTPRequestHandler]:
    class RunApiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                payload = _dispatch_get(store, self.path)
            except RunApiRouteError as exc:
                _write_json(
                    self,
                    exc.status,
                    {"error": {"code": exc.code, "message": exc.message}},
                )
                return
            except Exception as exc:
                _write_json(
                    self,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": {
                            "code": "internal_error",
                            "message": str(exc) or exc.__class__.__name__,
                        }
                    },
                )
                return
            _write_json(self, HTTPStatus.OK, payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    return RunApiHandler


class RunApiRouteError(Exception):
    """HTTP routing failure with response metadata."""

    def __init__(self, *, status: HTTPStatus, code: str, message: str) -> None:
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)


def _dispatch_get(store: LocalRunStore, request_path: str) -> dict[str, object]:
    path = urlsplit(request_path).path.strip("/")
    if path == "health":
        return {"status": "ok"}
    if not path:
        return {
            "name": "anexus-run-api",
            "status": "ok",
            "endpoints": [
                "/health",
                "/runs/{run_id}",
                "/runs/{run_id}/inspect",
                "/runs/{run_id}/events",
                "/runs/{run_id}/attempts",
                "/runs/{run_id}/live",
                "/runs/{run_id}/summary",
            ],
        }

    parts = path.split("/")
    if len(parts) not in {2, 3} or parts[0] != "runs":
        raise RunApiRouteError(
            status=HTTPStatus.NOT_FOUND,
            code="not_found",
            message="unknown endpoint",
        )

    run_id = unquote(parts[1]).strip()
    if not run_id:
        raise RunApiRouteError(
            status=HTTPStatus.BAD_REQUEST,
            code="invalid_run_id",
            message="run_id is required",
        )

    action = "inspect" if len(parts) == 2 else parts[2]
    if action == "inspect":
        return build_run_inspection(store, run_id=run_id)
    if action == "events":
        return build_run_events_payload(store, run_id=run_id)
    if action == "attempts":
        return build_run_attempts_payload(store, run_id=run_id)
    if action == "live":
        return build_run_live_payload(store, run_id=run_id)
    if action == "summary":
        return build_run_summary_payload(store, run_id=run_id)

    raise RunApiRouteError(
        status=HTTPStatus.NOT_FOUND,
        code="not_found",
        message="unknown run endpoint",
    )


def _write_json(
    handler: BaseHTTPRequestHandler,
    status: HTTPStatus,
    payload: dict[str, Any],
) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    handler.send_response(status.value)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)
