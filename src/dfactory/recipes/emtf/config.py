"""Configuration owned by the built-in EMTF recipe."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from dfactory.components.block_diffusion import SlidingWindowAttentionConfig


@dataclass(frozen=True)
class EMTFConfig:
    block_size: int = 32
    max_stages: int = 4
    confidence_threshold: float = 0.5
    force_mask_token_ids: tuple[int, ...] = ()
    sliding_window: SlidingWindowAttentionConfig | None = None

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> EMTFConfig:
        unknown = sorted(set(values) - {item.name for item in fields(cls)})
        if unknown:
            raise ValueError(f"unknown EMTF parameter(s): {', '.join(unknown)}")
        normalized = dict(values)
        if "force_mask_token_ids" in normalized:
            if not isinstance(normalized["force_mask_token_ids"], list | tuple):
                raise ValueError("recipe.params.force_mask_token_ids must be a sequence")
            normalized["force_mask_token_ids"] = tuple(normalized["force_mask_token_ids"])
        if isinstance(normalized.get("sliding_window"), dict):
            normalized["sliding_window"] = SlidingWindowAttentionConfig.from_mapping(normalized["sliding_window"])
        elif normalized.get("sliding_window") is not None:
            raise ValueError("recipe.params.sliding_window must be a mapping or null")
        try:
            config = cls(**normalized)
        except TypeError as exc:
            raise ValueError(f"invalid EMTF config: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool) or self.block_size <= 0:
            raise ValueError("recipe.params.block_size must be positive")
        if not isinstance(self.max_stages, int) or isinstance(self.max_stages, bool) or self.max_stages <= 0:
            raise ValueError("recipe.params.max_stages must be positive")
        if (
            not isinstance(self.confidence_threshold, int | float)
            or isinstance(self.confidence_threshold, bool)
            or not 0.0 <= self.confidence_threshold <= 1.0
        ):
            raise ValueError("recipe.params.confidence_threshold must be between 0 and 1")
        if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in self.force_mask_token_ids):
            raise ValueError("recipe.params.force_mask_token_ids must contain integers")
        if self.sliding_window is not None:
            self.sliding_window.validate()


__all__ = ["EMTFConfig"]
