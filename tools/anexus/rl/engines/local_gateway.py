"""Local OpenAI-compatible HTTP gateway for RL engines."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from anexus.gateway.failures import (
    classify_gateway_http_failure,
    classify_gateway_transport_failure,
    gateway_failure_payload,
)
from anexus.gateway.format import (
    anthropic_to_openai_chat,
    openai_chat_to_anthropic_message,
    wrap_anthropic_message_as_sse,
    wrap_openai_chat_as_sse,
)
from anexus.gateway.protocols import RouteDecision, RouteTarget
from anexus.gateway.routing import GatewayMirror, ModelRouter
from anexus.gateway.upstream import UPSTREAM_ROUTE_HEADER
from anexus.rl.engines.base import InferenceEngine

SESSION_HEADER = "X-Anexus-Session-ID"


class AsyncLocalAgentGateway:
    """Async local gateway exposing OpenAI-compatible chat completions.

    The service accepts standard `/v1/chat/completions` and compatible
    `/chat/completions` requests. Session identity is supplied through
    `X-Anexus-Session-ID` when needed instead of being embedded in the URL.
    """

    def __init__(
        self,
        *,
        engine: InferenceEngine,
        router: ModelRouter,
        session_id: str | None = None,
        upstream_proxy_url: str | None = None,
        upstream_route_header: str = UPSTREAM_ROUTE_HEADER,
        timeout_seconds: float = 7200.0,
        trace_dir: str | Path | None = None,
    ) -> None:
        """Initialize one local gateway around an RL inference engine."""
        self.engine = engine
        self.router = router
        self.session_id = session_id or engine.get_session_id()
        self.upstream_proxy_url = upstream_proxy_url.rstrip("/") if upstream_proxy_url else ""
        self.upstream_route_header = upstream_route_header
        self.timeout_seconds = timeout_seconds
        self._trace_dir = Path(trace_dir) if trace_dir else None
        self._trace_lock = asyncio.Lock()
        self._trace_files: dict[str, Path] = {}
        self._call_count = 0
        self._client: httpx.AsyncClient | None = None
        self._runner = None
        self._port: int | None = None

    @property
    def port(self) -> int | None:
        """Return the assigned listening port after startup."""
        return self._port

    @property
    def base_url(self) -> str:
        """Return the local gateway origin after startup."""
        if self._port is None:
            raise RuntimeError("async local agent gateway is not started")
        return f"http://127.0.0.1:{self._port}"

    async def start(self) -> int:
        """Start the gateway, return the listening port."""
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/{path:.*}", self._handle_post)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/", self._handle_health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 0)
        await site.start()
        self._port = site._server.sockets[0].getsockname()[1]
        self._runner = runner
        self._client = httpx.AsyncClient(timeout=self.timeout_seconds, verify=False)
        return self._port

    async def stop(self) -> None:
        """Close HTTP clients and stop the local server."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._port = None

    async def _handle_health(self, request) -> Any:
        from aiohttp import web

        return web.json_response({"status": "ok", "session_id": self.session_id})

    async def _handle_post(self, request) -> Any:
        from aiohttp import web

        body = await request.json()
        path_parts = _path_parts(request.path)
        session_id = request.headers.get(SESSION_HEADER, "") or _session_from_path(path_parts, self.session_id)

        if _is_anthropic_messages_path(path_parts):
            if path_parts and path_parts[-1] == "count_tokens":
                return web.json_response(_count_tokens(body))
            return await self._handle_anthropic(request, body, session_id)
        if _is_openai_chat_path(path_parts):
            return await self._handle_openai(request, body, session_id)
        return web.json_response({"error": "unsupported gateway path"}, status=404)

    async def _handle_openai(self, request, body: dict[str, Any], session_id: str) -> Any:
        from aiohttp import web

        wants_stream = bool(body.pop("stream", False))
        body.pop("stream_options", None)
        model = str(body.get("model", ""))

        decision = self._route_model(model)
        if decision.target == RouteTarget.UNAVAILABLE:
            return web.json_response(
                {"error": decision.reason, "type": "gateway_unavailable"},
                status=503,
            )

        t0 = time.time()
        if decision.target == RouteTarget.ENGINE:
            response = await self._forward_to_engine(body, session_id)
        else:
            if wants_stream and self._client and self.upstream_proxy_url:
                return await self._stream_mirror_passthrough(body, decision, request)
            response = await self._forward_to_mirror(body, decision)
            if decision.mirror and "error" not in response:
                mirror_name = getattr(decision.mirror, "name", "")
                if mirror_name:
                    self.router.record_success(mirror_name)
        latency_ms = (time.time() - t0) * 1000

        api_key = self._extract_api_key(request)
        await self._record_trace(api_key, body, response, protocol="openai", latency_ms=latency_ms)

        if "error" in response:
            return web.json_response(response, status=int(response.get("status_code", 502)))
        if wants_stream:
            return self._send_sse_response(wrap_openai_chat_as_sse(response))
        return web.json_response(response)

    async def _handle_anthropic(self, request, body: dict[str, Any], session_id: str) -> Any:
        from aiohttp import web

        wants_stream = bool(body.get("stream", False))
        openai_request = anthropic_to_openai_chat(body)
        model = str(openai_request.get("model", ""))

        decision = self._route_model(model)
        if decision.target == RouteTarget.UNAVAILABLE:
            return web.json_response(
                {"error": decision.reason, "type": "gateway_unavailable"},
                status=503,
            )

        t0 = time.time()
        if decision.target == RouteTarget.ENGINE:
            response = await self._forward_to_engine(openai_request, session_id)
        else:
            response = await self._forward_to_mirror(openai_request, decision)
            if decision.mirror and "error" not in response:
                mirror_name = getattr(decision.mirror, "name", "")
                if mirror_name:
                    self.router.record_success(mirror_name)
        latency_ms = (time.time() - t0) * 1000

        api_key = self._extract_api_key(request)
        await self._record_trace(api_key, body, response, protocol="anthropic", latency_ms=latency_ms)

        if "error" in response:
            return web.json_response(response, status=int(response.get("status_code", 502)))
        message = openai_chat_to_anthropic_message(response, original_request=body)
        if wants_stream:
            return self._send_sse_response(wrap_anthropic_message_as_sse(message))
        return web.json_response(message)

    def _route_model(self, model: str) -> RouteDecision:
        """Return the single configured router's decision for a model."""
        return self.router.route(model)

    async def _forward_to_engine(self, body: dict[str, Any], session_id: str) -> dict[str, Any]:
        url = self.engine.get_completion_url(session_id)
        if self._client is None:
            return {"error": "gateway client not initialized", "status_code": 503}
        try:
            resp = await self._client.post(url, json=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            failure = classify_gateway_http_failure(
                phase="gateway.engine.forward", status_code=exc.response.status_code
            )
            return gateway_failure_payload(
                failure,
                message=f"engine request failed with HTTP {exc.response.status_code}",
                body=exc.response.text,
            )
        except (httpx.ConnectError, httpx.ReadError, httpx.WriteError, OSError) as exc:
            failure = classify_gateway_transport_failure(
                phase="gateway.engine.forward", exception_type=exc.__class__.__name__
            )
            return gateway_failure_payload(
                failure,
                message=f"engine request failed: {exc}",
                exception_type=exc.__class__.__name__,
            )

    async def _forward_to_mirror(self, body: dict[str, Any], decision: RouteDecision) -> dict[str, Any]:
        mirror = decision.mirror
        if not self.upstream_proxy_url or mirror is None:
            return {
                "error": f"no upstream route for model {decision.model}",
                "phase": "gateway.mirror.forward",
                "reason": "mirror_route_missing",
                "retryable": False,
                "status_code": 502,
            }
        mirror_obj = mirror if isinstance(mirror, GatewayMirror) else None
        route_key = f"{mirror_obj.route_label}-openai" if mirror_obj else ""
        url = f"{self.upstream_proxy_url}/chat/completions"
        if self._client is None:
            return {"error": "gateway client not initialized", "status_code": 503}
        try:
            resp = await self._client.post(
                url,
                json=body,
                headers={self.upstream_route_header: route_key},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            mirror_name = mirror_obj.name if mirror_obj else ""
            if mirror_name:
                self.router.record_failure(mirror_name)
            failure = classify_gateway_http_failure(
                phase="gateway.mirror.forward", status_code=exc.response.status_code
            )
            return gateway_failure_payload(
                failure,
                message=f"upstream proxy request failed with HTTP {exc.response.status_code}",
                body=exc.response.text,
            )
        except (httpx.ConnectError, httpx.ReadError, httpx.WriteError, OSError) as exc:
            mirror_name = getattr(mirror, "name", "") if mirror else ""
            if mirror_name:
                self.router.record_failure(mirror_name)
            failure = classify_gateway_transport_failure(
                phase="gateway.mirror.forward", exception_type=exc.__class__.__name__
            )
            return gateway_failure_payload(
                failure,
                message=f"upstream proxy request failed: {exc}",
                exception_type=exc.__class__.__name__,
            )

    async def _stream_mirror_passthrough(self, body: dict[str, Any], decision: RouteDecision, request) -> Any:
        """Forward SSE chunks as the mirror produces them."""
        from aiohttp import web

        mirror = decision.mirror
        mirror_obj = mirror if isinstance(mirror, GatewayMirror) else None
        route_key = f"{mirror_obj.route_label}-openai" if mirror_obj else ""
        url = f"{self.upstream_proxy_url}/chat/completions"
        body["stream"] = True

        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await response.prepare(request)

        if self._client is None:
            await response.write(b'data: {"error": "client not ready"}\n\n')
            return response

        async with self._client.stream(
            "POST",
            url,
            json=body,
            headers={self.upstream_route_header: route_key},
        ) as upstream:
            async for chunk in upstream.aiter_bytes():
                await response.write(chunk)

        return response

    def _send_sse_response(self, chunks: Any) -> Any:
        from aiohttp import web

        body = "".join(chunks).encode("utf-8")
        return web.Response(
            body=body,
            status=200,
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    def _extract_api_key(self, request) -> str:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        return request.headers.get("x-api-key", "").strip()

    async def _record_trace(
        self,
        api_key: str,
        request_data: dict[str, Any],
        response_data: dict[str, Any],
        *,
        protocol: str = "openai",
        latency_ms: float = 0,
    ) -> None:
        if self._trace_dir is None:
            return
        key_name = api_key.strip() or "default"
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self._trace_dir / f"{key_name}.jsonl"

        async with self._trace_lock:
            self._call_count += 1
            self._trace_files[key_name] = trace_path

        usage = response_data.get("usage", {})
        model = str(response_data.get("model", request_data.get("model", "")))
        entry = {
            "timestamp": time.time(),
            "call_index": self._call_count,
            "protocol": protocol,
            "model_requested": str(request_data.get("model", "")),
            "model_resolved": model,
            "input_tokens": int(usage.get("prompt_tokens", usage.get("input_tokens", 0))),
            "output_tokens": int(usage.get("completion_tokens", usage.get("output_tokens", 0))),
            "latency_ms": int(latency_ms),
            "status": "ok" if "error" not in response_data else "error",
        }
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
        async with self._trace_lock:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(line)

    @property
    def trace_files(self) -> dict[str, Path]:
        """Return trace files keyed by gateway API key."""
        return dict(self._trace_files)

    async def __aenter__(self) -> AsyncLocalAgentGateway:
        """Start and return the gateway as an async context manager."""
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Stop the gateway when leaving the async context."""
        await self.stop()


# ──── Helpers ──────────────────────────────────────────────────────────


def _path_parts(path: str) -> list[str]:
    return [part for part in path.split("?", 1)[0].strip("/").split("/") if part]


def _is_openai_chat_path(parts: list[str]) -> bool:
    return "chat" in parts and "completions" in parts


def _is_anthropic_messages_path(parts: list[str]) -> bool:
    return "v1" in parts and "messages" in parts


def _session_from_path(parts: list[str], default: str) -> str:
    if not parts:
        return default
    if parts[0] == "v1":
        if len(parts) >= 2 and parts[1] != "chat":
            return parts[1]
        return default
    return parts[0]


def _count_tokens(request_data: dict[str, Any]) -> dict[str, int]:
    text = _flatten_text(request_data.get("system"))
    for message in request_data.get("messages", []):
        if isinstance(message, dict):
            text += "\n" + _flatten_text(message.get("content"))
    return {"input_tokens": max(1, len(text) // 4)}


def _flatten_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_flatten_text(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") == "text":
            return str(value.get("text", ""))
        if "content" in value:
            return _flatten_text(value["content"])
    return str(value)
