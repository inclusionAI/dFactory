"""Configuration owned by ESTF continued pretraining."""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Real
from typing import Any, Literal


@dataclass(frozen=True)
class ESTFConfig:
    block_size: int = 32
    max_stages: int = 4
    confidence_threshold: float = 0.85
    confidence_source: Literal["ground_truth", "maximum"] = "ground_truth"
    selective_loss: bool = False
    initialize_random_known: bool = False
    initial_known_ratio: tuple[float, float] = (0.2, 0.2)
    initial_known_schedule_power: float = 1.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ESTFConfig:
        unknown = sorted(set(values) - {item.name for item in fields(cls)})
        if unknown:
            raise ValueError(f"unknown ESTF parameter(s): {', '.join(unknown)}")
        normalized = dict(values)
        if "initial_known_ratio" in normalized:
            try:
                normalized["initial_known_ratio"] = tuple(normalized["initial_known_ratio"])
            except TypeError as exc:
                raise ValueError("recipe.params.initial_known_ratio must contain two values") from exc
        try:
            config = cls(**normalized)
        except TypeError as exc:
            raise ValueError(f"invalid ESTF config: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool) or self.block_size <= 0:
            raise ValueError("recipe.params.block_size must be a positive integer")
        if not isinstance(self.max_stages, int) or isinstance(self.max_stages, bool) or self.max_stages <= 0:
            raise ValueError("recipe.params.max_stages must be a positive integer")
        if (
            not isinstance(self.confidence_threshold, int | float)
            or isinstance(self.confidence_threshold, bool)
            or not 0.0 <= self.confidence_threshold <= 1.0
        ):
            raise ValueError("recipe.params.confidence_threshold must be between 0 and 1")
        if self.confidence_source not in {"ground_truth", "maximum"}:
            raise ValueError("recipe.params.confidence_source must be 'ground_truth' or 'maximum'")
        if not isinstance(self.selective_loss, bool):
            raise ValueError("recipe.params.selective_loss must be a boolean")
        if not isinstance(self.initialize_random_known, bool):
            raise ValueError("recipe.params.initialize_random_known must be a boolean")
        if len(self.initial_known_ratio) != 2:
            raise ValueError("recipe.params.initial_known_ratio must contain two values")
        low, high = self.initial_known_ratio
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in self.initial_known_ratio):
            raise ValueError("recipe.params.initial_known_ratio values must be numbers")
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("recipe.params.initial_known_ratio must satisfy 0 <= low <= high <= 1")
        if (
            not isinstance(self.initial_known_schedule_power, Real)
            or isinstance(self.initial_known_schedule_power, bool)
            or self.initial_known_schedule_power <= 0
        ):
            raise ValueError("recipe.params.initial_known_schedule_power must be positive")


__all__ = ["ESTFConfig"]
