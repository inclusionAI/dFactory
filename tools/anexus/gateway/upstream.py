"""Host-side routed upstream proxy for gateway mirrors."""

from __future__ import annotations

import ssl
import threading
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlsplit, urlunparse
from urllib.request import Request, urlopen

from anexus.gateway.failures import classify_upstream_proxy_failure

UPSTREAM_ROUTE_HEADER = "X-Anexus-Upstream-Route"
_CHUNK_SIZE = 16 * 1024
_CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError)
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


@dataclass
class GatewayUpstreamProxy:
    """Running routed upstream proxy."""

    server: ThreadingHTTPServer
    thread: threading.Thread
    routes: dict[str, str]
    events: deque[dict[str, object]]
    event_lock: threading.Lock

    @property
    def base_url(self) -> str:
        """Return the host-visible HTTP origin."""
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    @property
    def tunnel_upstream(self) -> str:
        """Return the host and port consumed by the AKernel tunnel."""
        host, port = self.server.server_address
        return f"{host}:{port}"

    def shutdown(self) -> None:
        """Stop the proxy server and join its serving thread."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def event_snapshot(self) -> list[dict[str, object]]:
        """Return a stable copy of buffered proxy events."""
        with self.event_lock:
            return [dict(event) for event in self.events]

    def __enter__(self) -> GatewayUpstreamProxy:
        """Return this running proxy as a context manager."""
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        """Shut down the proxy when leaving the context."""
        self.shutdown()


@dataclass(frozen=True)
class _RequestBodyReadError:
    message: str
    error_type: str
    client_disconnected: bool = False
    status: int | None = 400


def start_gateway_upstream_proxy(
    routes: dict[str, str],
    *,
    route_header: str = UPSTREAM_ROUTE_HEADER,
    timeout_seconds: float,
    event_buffer_size: int,
    tls_verify: bool,
) -> GatewayUpstreamProxy:
    """Start a localhost proxy that routes by an internal header key."""
    normalized = _normalize_routes(routes)
    if not normalized:
        raise ValueError("upstream proxy routes must not be empty")
    if event_buffer_size < 1:
        raise ValueError("event_buffer_size must be >= 1")

    events: deque[dict[str, object]] = deque(maxlen=event_buffer_size)
    event_lock = threading.Lock()
    handler = _make_proxy_handler(
        routes=normalized,
        route_header=route_header,
        timeout_seconds=timeout_seconds,
        tls_verify=tls_verify,
        events=events,
        event_lock=event_lock,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=server.serve_forever,
        name="anexus-upstream-proxy",
        daemon=True,
    )
    thread.start()
    return GatewayUpstreamProxy(
        server=server,
        thread=thread,
        routes=normalized,
        events=events,
        event_lock=event_lock,
    )


def _make_proxy_handler(
    *,
    routes: dict[str, str],
    route_header: str,
    timeout_seconds: float,
    tls_verify: bool,
    events: deque[dict[str, object]],
    event_lock: threading.Lock,
) -> type[BaseHTTPRequestHandler]:
    request_seq = 0

    def next_request_id() -> int:
        nonlocal request_seq
        with event_lock:
            request_seq += 1
            return request_seq

    def emit_proxy_event(kind: str, **payload: object) -> None:
        payload = _with_failure_classification(kind, payload)
        with event_lock:
            events.append({"kind": kind, **payload})

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._forward()

        def do_POST(self) -> None:
            self._forward()

        def do_PUT(self) -> None:
            self._forward()

        def do_DELETE(self) -> None:
            self._forward()

        def _forward(self) -> None:
            request_id = next_request_id()
            route_key = (self.headers.get(route_header) or "").strip().lower()
            upstream_base = routes.get(route_key)
            if upstream_base is None:
                emit_proxy_event(
                    "route_not_found",
                    request_id=request_id,
                    route_key=route_key,
                    path=self.path,
                    method=self.command,
                    status=404,
                )
                self.send_error(404, "no upstream route matched")
                return

            target = urlsplit(self.path)
            upstream_url = _upstream_url(upstream_base, target.path or "/", target.query)
            body, body_error = _read_request_body(self)
            if body_error is not None:
                emit_proxy_event(
                    "request_body_error",
                    request_id=request_id,
                    route_key=route_key,
                    path=self.path,
                    method=self.command,
                    status=body_error.status or "",
                    error_type=body_error.error_type,
                    error=body_error.message,
                    client_disconnected=body_error.client_disconnected,
                )
                if body_error.status is not None:
                    self.send_error(body_error.status, body_error.message)
                return
            request = Request(upstream_url, data=body, method=self.command)
            _copy_request_headers(self, request, route_header=route_header)

            emit_proxy_event(
                "upstream_request",
                request_id=request_id,
                route_key=route_key,
                path=self.path,
                method=self.command,
                upstream_url=upstream_url,
                body_bytes=len(body or b""),
            )
            try:
                context = _ssl_context(upstream_url, tls_verify=tls_verify)
                response = urlopen(request, timeout=timeout_seconds, context=context)
                self._send_upstream_response(
                    response,
                    response.status,
                    request_id,
                    route_key,
                )
            except HTTPError as exc:
                self._send_upstream_response(exc, exc.code, request_id, route_key)
            except URLError as exc:
                emit_proxy_event(
                    "upstream_error",
                    request_id=request_id,
                    route_key=route_key,
                    path=self.path,
                    method=self.command,
                    upstream_url=upstream_url,
                    status=502,
                    error_type=exc.__class__.__name__,
                    error=str(exc.reason),
                )
                self.send_error(502, f"upstream request failed: {exc.reason}")

        def _send_upstream_response(
            self,
            response: object,
            status: int,
            request_id: int,
            route_key: str,
        ) -> None:
            bytes_sent = 0
            event_kind = "response_complete"
            client_disconnected = False
            error_type = ""
            response_error = ""
            upstream_close_error = ""
            try:
                emit_proxy_event(
                    "upstream_response",
                    request_id=request_id,
                    route_key=route_key,
                    path=self.path,
                    method=self.command,
                    status=status,
                )
                self.send_response(status)
                headers = getattr(response, "headers", {})
                for key, value in headers.items():
                    key_lower = key.lower()
                    if key_lower in _HOP_BY_HOP_HEADERS:
                        continue
                    self.send_header(key, value)
                self.end_headers()
                while True:
                    chunk = response.read(_CHUNK_SIZE)  # type: ignore[attr-defined]
                    if not chunk:
                        break
                    bytes_sent += len(chunk)
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except _CLIENT_DISCONNECT_ERRORS as exc:
                        event_kind = "stream_error"
                        client_disconnected = True
                        error_type = exc.__class__.__name__
                        response_error = _format_exception(exc)
                        break
            except _CLIENT_DISCONNECT_ERRORS as exc:
                event_kind = "stream_error"
                client_disconnected = True
                error_type = exc.__class__.__name__
                response_error = _format_exception(exc)
            except (OSError, ValueError, UnicodeDecodeError, RuntimeError) as exc:
                event_kind = "stream_error"
                error_type = exc.__class__.__name__
                response_error = _format_exception(exc)
            finally:
                upstream_close_error = _close_upstream_response(response)
            emit_proxy_event(
                event_kind,
                request_id=request_id,
                route_key=route_key,
                path=self.path,
                method=self.command,
                status=status,
                error_type=error_type,
                client_disconnected=client_disconnected,
                response_error=response_error,
                upstream_close_error=upstream_close_error,
                bytes_sent=bytes_sent,
            )

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def _ssl_context(url: str, *, tls_verify: bool) -> ssl.SSLContext | None:
    if not url.startswith("https") or tls_verify:
        return None
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _normalize_routes(routes: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for route_key, upstream_url in routes.items():
        if not route_key or not upstream_url:
            continue
        parsed = urlparse(upstream_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid upstream URL for route {route_key!r}: {upstream_url!r}")
        normalized[route_key.lower()] = upstream_url.rstrip("/")
    return normalized


def _upstream_url(upstream_base: str, request_path: str, query: str) -> str:
    parsed_base = urlparse(upstream_base)
    return urlunparse(
        (
            parsed_base.scheme,
            parsed_base.netloc,
            request_path,
            "",
            query,
            "",
        )
    )


def _read_request_body(
    handler: BaseHTTPRequestHandler,
) -> tuple[bytes | None, _RequestBodyReadError | None]:
    raw_content_length = handler.headers.get("Content-Length", "")
    if raw_content_length == "":
        return None, None
    if not isinstance(raw_content_length, str) or not raw_content_length.strip().isdigit():
        return None, _RequestBodyReadError(
            message=f"invalid Content-Length header: {raw_content_length!r}",
            error_type="ValueError",
        )
    content_length = int(raw_content_length)
    if content_length < 0:
        return None, _RequestBodyReadError(
            message=f"negative Content-Length header: {raw_content_length!r}",
            error_type="ValueError",
        )
    if content_length <= 0:
        return None, None
    try:
        body = handler.rfile.read(content_length)
    except _CLIENT_DISCONNECT_ERRORS as exc:
        return None, _RequestBodyReadError(
            message=str(exc) or "client disconnected while sending request body",
            error_type=exc.__class__.__name__,
            client_disconnected=True,
            status=None,
        )
    except OSError as exc:
        return None, _RequestBodyReadError(
            message=f"request body read failed: {exc}",
            error_type=exc.__class__.__name__,
        )
    if len(body) < content_length:
        return None, _RequestBodyReadError(
            message=(f"request body ended early: expected {content_length} bytes, got {len(body)}"),
            error_type="IncompleteRead",
            client_disconnected=True,
            status=None,
        )
    return body, None


def _copy_request_headers(
    handler: BaseHTTPRequestHandler,
    request: Request,
    *,
    route_header: str,
) -> None:
    internal_header = route_header.lower()
    target_netloc = urlparse(request.full_url).netloc
    for key, value in handler.headers.items():
        key_lower = key.lower()
        if key_lower in _HOP_BY_HOP_HEADERS or key_lower == internal_header or key_lower == "accept-encoding":
            continue
        request.add_header(key, value)
    request.add_header("Accept-Encoding", "identity")
    request.add_header("Host", target_netloc)


def _format_exception(exc: BaseException) -> str:
    message = str(exc)
    if message:
        return f"{exc.__class__.__name__}: {message}"
    return exc.__class__.__name__


def _close_upstream_response(response: object) -> str:
    close = getattr(response, "close", None)
    if not callable(close):
        return ""
    try:
        close()
    except OSError as exc:
        return _format_exception(exc)
    return ""


def _with_failure_classification(
    kind: str,
    payload: dict[str, object],
) -> dict[str, object]:
    if kind not in {
        "malformed_request",
        "request_body_error",
        "route_not_found",
        "response_complete",
        "stream_error",
        "upstream_error",
        "upstream_response",
    }:
        return payload
    classification = classify_upstream_proxy_failure(
        kind=kind,
        error_type=str(payload.get("error_type") or ""),
        error=str(payload.get("error") or ""),
        status=None if kind == "response_complete" else payload.get("status"),
        client_disconnected=bool(payload.get("client_disconnected") or False),
        response_error=str(payload.get("response_error") or ""),
        upstream_close_error=str(payload.get("upstream_close_error") or ""),
    )
    if classification is None:
        return payload
    return {
        **payload,
        "failure_reason": classification.reason,
        "retryable": classification.retryable,
    }
