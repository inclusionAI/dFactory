"""Model parallelization and activation contexts backed by VeOmni."""

from __future__ import annotations

from types import MethodType
from typing import Any

from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.torch_parallelize import build_parallelize_model

from dfactory.config import DistributedConfig, ModelConfig


def parallelize(
    model: Any,
    model_config: ModelConfig,
    distributed: DistributedConfig,
    *,
    sharding_plan: Any = None,
) -> Any:
    if sharding_plan is not None:
        model.get_parallel_plan = MethodType(lambda _model: sharding_plan, model)
    return build_parallelize_model(
        model,
        weights_path=model_config.weights_path,
        enable_full_shard=True,
        enable_mixed_precision=True,
        enable_gradient_checkpointing=distributed.gradient_checkpointing,
        basic_modules=getattr(model, "_no_split_modules", None),
        init_device=model_config.init_device,
        broadcast_model_weights_from_rank0=True,
    )


def activation_contexts(distributed: DistributedConfig) -> tuple[Any, Any]:
    return build_activation_offloading_context(
        enable_activation_offload=distributed.activation_offload,
        enable_gradient_checkpointing=distributed.gradient_checkpointing,
    )


__all__ = ["activation_contexts", "parallelize"]
