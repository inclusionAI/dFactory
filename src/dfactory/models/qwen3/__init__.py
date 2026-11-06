"""Qwen3 Block Diffusion provider and model classes."""

from transformers import Qwen3Config

from dfactory.models.qwen3.provider import Qwen3Provider
from dfactory.models.spec import ModelSpec

model_spec = ModelSpec(
    name="qwen3",
    provider_type=Qwen3Provider,
    description="Qwen3 Block Diffusion with native Qwen3 configuration and external masks.",
)


def __getattr__(name):
    if name in {"Qwen3ForBDLM", "Qwen3Model", "Qwen3PreTrainedModel"}:
        from dfactory.models.qwen3 import modeling

        return getattr(modeling, name)
    raise AttributeError(name)


__all__ = ["Qwen3Config", "Qwen3ForBDLM", "Qwen3Model", "Qwen3PreTrainedModel", "Qwen3Provider", "model_spec"]
