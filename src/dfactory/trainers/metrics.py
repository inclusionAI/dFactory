"""Trainer metric values and aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StepMetrics:
    """Metrics produced by one trainer micro-batch."""

    values: Mapping[str, float] = field(default_factory=dict)
    weight: float = 1.0


class MetricAccumulator:
    def __init__(self) -> None:
        self._weighted_values: dict[str, float] = defaultdict(float)
        self._weights: dict[str, float] = defaultdict(float)

    def add(self, metrics: StepMetrics) -> None:
        if metrics.weight < 0:
            raise ValueError("metric weight must not be negative")
        for name, value in metrics.values.items():
            self._weighted_values[name] += float(value) * metrics.weight
            self._weights[name] += metrics.weight

    def compute(self) -> dict[str, float]:
        return {
            name: value / self._weights[name]
            for name, value in self._weighted_values.items()
            if self._weights[name] > 0
        }


__all__ = ["StepMetrics"]
