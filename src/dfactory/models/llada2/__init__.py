"""LLaDA2 model provider."""

from dfactory.models.llada2.configuration import LLaDA2MoeConfig
from dfactory.models.llada2.modeling import LLaDA2MoeModel, LLaDA2MoeModelLM
from dfactory.models.llada2.provider import LLaDA2Provider
from dfactory.models.spec import ModelSpec

model_spec = ModelSpec(
    name="llada2",
    provider_type=LLaDA2Provider,
    description="LLaDA2 Mixture-of-Experts models.",
)

__all__ = ["LLaDA2MoeConfig", "LLaDA2MoeModel", "LLaDA2MoeModelLM", "LLaDA2Provider", "model_spec"]
