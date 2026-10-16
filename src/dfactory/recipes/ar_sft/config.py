"""Configuration owned by the built-in autoregressive SFT recipe."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ARSFTConfig:
    """Autoregressive SFT has no algorithm parameters in its first public slice."""

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ARSFTConfig:
        unknown = sorted(set(values) - {item.name for item in fields(cls)})
        if unknown:
            raise ValueError(f"unknown AR SFT parameter(s): {', '.join(unknown)}")
        return cls(**values)


__all__ = ["ARSFTConfig"]
