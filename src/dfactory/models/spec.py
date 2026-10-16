"""Model-provider metadata."""

from dataclasses import dataclass

from dfactory.models.base import ModelProvider


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider_type: type[ModelProvider]
    description: str


__all__ = ["ModelSpec"]
