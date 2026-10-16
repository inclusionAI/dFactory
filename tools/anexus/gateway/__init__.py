"""Gateway session routing."""

from anexus.gateway.activity import GatewayActivityMonitor
from anexus.gateway.failures import (
    GatewayFailure,
    UpstreamProxyFailure,
    classify_gateway_http_failure,
    classify_gateway_transport_failure,
    classify_upstream_proxy_failure,
    gateway_failure_affects_task,
    gateway_failure_payload,
)
from anexus.gateway.format import (
    anthropic_to_openai_chat,
    openai_chat_to_anthropic_message,
    wrap_anthropic_message_as_sse,
    wrap_openai_chat_as_sse,
)
from anexus.gateway.health import GatewayHealthReport, check_gateway_session
from anexus.gateway.kgb import KGBGatewaySession, KGBPool
from anexus.gateway.protocols import (
    RouteDecision,
    RouteTarget,
)
from anexus.gateway.routing import (
    CircuitBreaker,
    GatewayMirror,
    GatewayProtocol,
    ModelRouter,
    alias_target_model,
    build_mirror_routes,
    model_alias,
)
from anexus.gateway.session import (
    GatewayActivitySnapshot,
    GatewaySession,
    GatewaySessionProvider,
    KGBGatewaySessionProvider,
    KGBSessionConfig,
    NoopGatewaySession,
    NoopGatewaySessionProvider,
)
from anexus.gateway.upstream import (
    UPSTREAM_ROUTE_HEADER,
    GatewayUpstreamProxy,
    start_gateway_upstream_proxy,
)

__all__ = [
    "CircuitBreaker",
    "GatewayMirror",
    "GatewayFailure",
    "GatewayActivityMonitor",
    "GatewayActivitySnapshot",
    "GatewayHealthReport",
    "GatewayProtocol",
    "GatewaySession",
    "GatewaySessionProvider",
    "KGBGatewaySession",
    "KGBGatewaySessionProvider",
    "KGBPool",
    "KGBSessionConfig",
    "ModelRouter",
    "NoopGatewaySession",
    "NoopGatewaySessionProvider",
    "RouteDecision",
    "RouteTarget",
    "UPSTREAM_ROUTE_HEADER",
    "UpstreamProxyFailure",
    "GatewayUpstreamProxy",
    "alias_target_model",
    "anthropic_to_openai_chat",
    "build_mirror_routes",
    "check_gateway_session",
    "classify_gateway_http_failure",
    "classify_gateway_transport_failure",
    "classify_upstream_proxy_failure",
    "gateway_failure_affects_task",
    "gateway_failure_payload",
    "model_alias",
    "openai_chat_to_anthropic_message",
    "start_gateway_upstream_proxy",
    "wrap_anthropic_message_as_sse",
    "wrap_openai_chat_as_sse",
]
