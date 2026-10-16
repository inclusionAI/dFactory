"""Configuration owned by the built-in SFT recipe."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from dfactory.components.block_diffusion import SlidingWindowAttentionConfig


@dataclass(frozen=True)
class SFTConfig:
    block_size: int = 32
    noise_range: tuple[float, float] = (0.3, 0.8)
    same_token_labels: bool = True
    force_mask_token_ids: tuple[int, ...] = ()
    sliding_window: SlidingWindowAttentionConfig | None = None

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> SFTConfig:
        unknown = sorted(set(values) - {item.name for item in fields(cls)})
        if unknown:
            raise ValueError(f"unknown SFT parameter(s): {', '.join(unknown)}")
        normalized = dict(values)
        if "noise_range" in normalized:
            normalized["noise_range"] = tuple(normalized["noise_range"])
        if "force_mask_token_ids" in normalized:
            normalized["force_mask_token_ids"] = tuple(normalized["force_mask_token_ids"])
        if isinstance(normalized.get("sliding_window"), dict):
            normalized["sliding_window"] = SlidingWindowAttentionConfig.from_mapping(normalized["sliding_window"])
        elif normalized.get("sliding_window") is not None:
            raise ValueError("recipe.params.sliding_window must be a mapping or null")
        config = cls(**normalized)
        config.validate()
        return config

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError("recipe.params.block_size must be positive")
        if len(self.noise_range) != 2:
            raise ValueError("recipe.params.noise_range must contain two values")
        low, high = self.noise_range
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("recipe.params.noise_range must satisfy 0 <= low <= high <= 1")
        if self.sliding_window is not None:
            self.sliding_window.validate()


__all__ = ["SFTConfig"]
