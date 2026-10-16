"""Gateway routing data types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RouteTarget(StrEnum):
    """Identify the destination selected for an LLM request."""

    ENGINE = "engine"
    MIRROR = "mirror"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RouteDecision:
    """Result of model routing — where to send an LLM request."""

    target: RouteTarget
    model: str
    mirror: object | None = None
    is_fallback: bool = False
    is_default: bool = False
    reason: str = ""
