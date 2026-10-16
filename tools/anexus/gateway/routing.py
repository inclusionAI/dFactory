"""Gateway model routing and trace naming helpers."""

from __future__ import annotations

import time
from typing import Literal

from anexus.config.catalog import GatewayMirror
from anexus.gateway.protocols import RouteDecision, RouteTarget

GatewayProtocol = Literal["openai", "anthropic"]

PROTOCOLS: frozenset[GatewayProtocol] = frozenset({"openai", "anthropic"})
ANTHROPIC_ROLE_PREFIXES = frozenset({"main", "opus", "sonnet", "haiku", "small-fast"})


class ModelRouter:
    """Route engine and mirror models with per-mirror health tracking."""

    def __init__(
        self,
        *,
        mirrors: list[GatewayMirror] | None = None,
        engine_models: set[str] | None = None,
        strict: bool = False,
        fallback_mirror: GatewayMirror | None = None,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
    ) -> None:
        """Build one routing table and its mirror circuit breakers."""
        self._mirrors = mirrors or []
        self._engine_models = {model for model in (engine_models or set()) if model}
        self._model_to_mirror: dict[str, GatewayMirror] = {}
        self._mirror_health: dict[str, CircuitBreaker] = {}
        self._strict = strict
        self._fallback = fallback_mirror
        for mirror in self._mirrors:
            self._mirror_health[mirror.name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout_seconds=recovery_timeout_seconds,
            )
            for model in mirror.models:
                self._model_to_mirror.setdefault(model, mirror)
        if fallback_mirror and fallback_mirror.name not in self._mirror_health:
            self._mirror_health[fallback_mirror.name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout_seconds=recovery_timeout_seconds,
            )

    def is_engine_model(self, model: str) -> bool:
        """Return whether a model routes to the local inference engine."""
        if model in self._engine_models:
            return True
        if model in self._model_to_mirror:
            return False
        if self._strict:
            raise KeyError(f"model is not registered in engine models or mirrors: {model}")
        return True

    def mirror_for_model(self, model: str) -> GatewayMirror | None:
        """Return the configured mirror for a model."""
        return self._model_to_mirror.get(model)

    def mirror_route_key(self, model: str) -> str | None:
        """Return the upstream proxy route key for a model."""
        mirror = self.mirror_for_model(model)
        if mirror is None or mirror.protocol is None:
            return None
        return f"{mirror.route_label}-{mirror.protocol}"

    def mirror_protocol(self, model: str) -> GatewayProtocol | None:
        """Return the mirror protocol for a model."""
        mirror = self.mirror_for_model(model)
        return mirror.protocol if mirror else None

    def route(self, model: str) -> RouteDecision:
        """Return the current routing decision for a model."""
        if model in self._engine_models:
            return RouteDecision(target=RouteTarget.ENGINE, model=model)
        mirror = self._model_to_mirror.get(model)
        if mirror is not None:
            if self._mirror_health[mirror.name].is_open:
                fallback = self._fallback
                if fallback and not self._mirror_health[fallback.name].is_open:
                    return RouteDecision(
                        target=RouteTarget.MIRROR,
                        model=model,
                        mirror=fallback,
                        is_fallback=True,
                    )
                return RouteDecision(
                    target=RouteTarget.UNAVAILABLE,
                    model=model,
                    reason=f"mirror {mirror.name} circuit open, no fallback available",
                )
            return RouteDecision(target=RouteTarget.MIRROR, model=model, mirror=mirror)
        if self._strict:
            return RouteDecision(
                target=RouteTarget.UNAVAILABLE,
                model=model,
                reason=f"model {model} not registered in engine or mirrors",
            )
        return RouteDecision(target=RouteTarget.ENGINE, model=model, is_default=True)

    def record_success(self, mirror_name: str) -> None:
        """Reset health failures for a mirror after a successful request."""
        breaker = self._mirror_health.get(mirror_name)
        if breaker is not None:
            breaker.record_success()

    def record_failure(self, mirror_name: str) -> None:
        """Record a failed request against one mirror."""
        breaker = self._mirror_health.get(mirror_name)
        if breaker is not None:
            breaker.record_failure()


def model_alias(
    protocol: GatewayProtocol,
    model: str,
    alias_prefix: str | None = None,
    *,
    role: str | None = None,
) -> str:
    """Build a protocol-qualified gateway model alias."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported gateway protocol: {protocol}")
    parts = [protocol]
    if role:
        parts.append(role.strip("/"))
    parts.append(model.strip("/"))
    inner = "/".join(part for part in parts if part)
    prefix = (alias_prefix or "").strip("/")
    return f"{prefix}/{inner}" if prefix else inner


def alias_target_model(model: str) -> str:
    """Extract the upstream target model from a protocol-qualified alias."""
    parts = model.strip("/").split("/")
    for index, part in enumerate(parts):
        if part not in PROTOCOLS:
            continue
        target = parts[index + 1 :]
        if target and target[0] in ANTHROPIC_ROLE_PREFIXES:
            target = target[1:]
        return "/".join(target) or model
    return model


def build_mirror_routes(mirrors: list[GatewayMirror]) -> dict[str, str]:
    """Build route-key to upstream-url mappings for host-side proxy setup."""
    routes: dict[str, str] = {}
    for mirror in mirrors:
        if mirror.openai_upstream:
            routes[f"{mirror.route_label}-openai"] = mirror.openai_upstream.rstrip("/")
        if mirror.anthropic_upstream:
            routes[f"{mirror.route_label}-anthropic"] = mirror.anthropic_upstream.rstrip("/")
    return routes


# ──── Circuit Breaker ──────────────────────────────────────────────────


class CircuitBreaker:
    """Per-mirror circuit breaker with half-open recovery.

    States:
    - CLOSED: requests pass through, failures are counted
    - OPEN: requests are rejected immediately
    - HALF_OPEN: one probe request is allowed through
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize the failure threshold and recovery window."""
        self._failure_threshold = failure_threshold
        self._recovery_timeout_seconds = recovery_timeout_seconds
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._state: str = "closed"

    @property
    def is_open(self) -> bool:
        """Return whether requests are currently blocked."""
        if self._state == "closed":
            return False
        if self._state == "open":
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout_seconds:
                self._state = "half_open"
                return False
            return True
        return False

    def record_success(self) -> None:
        """Close the circuit after a successful request."""
        self._failure_count = 0
        self._state = "closed"

    def record_failure(self) -> None:
        """Record a failure and open the circuit at the threshold."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._state = "open"

    @property
    def state(self) -> str:
        """Return the current circuit state."""
        _ = self.is_open
        return self._state
