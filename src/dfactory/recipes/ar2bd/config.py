"""Configuration owned by autoregressive-to-Block-Diffusion pretraining."""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class AR2BDConfig:
    block_size: int = 32
    noise_range: tuple[float, float] = (0.3, 0.8)
    same_token_labels: bool = False

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> AR2BDConfig:
        unknown = sorted(set(values) - {item.name for item in fields(cls)})
        if unknown:
            raise ValueError(f"unknown AR2BD parameter(s): {', '.join(unknown)}")
        normalized = dict(values)
        if "noise_range" in normalized:
            try:
                normalized["noise_range"] = tuple(normalized["noise_range"])
            except TypeError as exc:
                raise ValueError("recipe.params.noise_range must contain two values") from exc
        try:
            config = cls(**normalized)
        except TypeError as exc:
            raise ValueError(f"invalid AR2BD config: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool) or self.block_size <= 0:
            raise ValueError("recipe.params.block_size must be a positive integer")
        if len(self.noise_range) != 2:
            raise ValueError("recipe.params.noise_range must contain two values")
        low, high = self.noise_range
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in self.noise_range):
            raise ValueError("recipe.params.noise_range values must be numbers")
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("recipe.params.noise_range must satisfy 0 <= low <= high <= 1")
        if not isinstance(self.same_token_labels, bool):
            raise ValueError("recipe.params.same_token_labels must be a boolean")


__all__ = ["AR2BDConfig"]
