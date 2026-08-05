from veomni.models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY

from .configuration_llada2_moe import LLaDA2MoeConfig
from .modeling_llada2_moe import LLaDA2MoeModel, LLaDA2MoeModelLM, LLaDA2MoePreTrainedModel


@MODEL_CONFIG_REGISTRY.register("llada2_moe_veomni")
def register_llada2_moe_config():
    return LLaDA2MoeConfig


@MODELING_REGISTRY.register("llada2_moe_veomni")
def register_llada2_moe_modeling(architecture: str):
    if architecture and ("ForCausalLM" in architecture or "ModelLM" in architecture):
        return LLaDA2MoeModelLM
    if architecture and "Model" in architecture:
        return LLaDA2MoeModel
    return LLaDA2MoeModelLM

ModelClass = LLaDA2MoeModelLM

__all__ = [
    "LLaDA2MoeConfig",
    "LLaDA2MoeModel",
    "LLaDA2MoeModelLM",
    "LLaDA2MoePreTrainedModel",
    "ModelClass",
]
