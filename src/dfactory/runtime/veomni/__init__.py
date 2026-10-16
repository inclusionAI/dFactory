"""The sole integration boundary between dFactory and VeOmni."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "VeOmniGateway":
        from dfactory.runtime.veomni.gateway import VeOmniGateway

        return VeOmniGateway
    raise AttributeError(name)


__all__ = ["VeOmniGateway"]
