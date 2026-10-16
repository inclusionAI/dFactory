"""Facade used by dFactory trainers to access VeOmni."""

from __future__ import annotations

from typing import Any

from dfactory.config import DistributedConfig, ModelConfig, OptimizationConfig
from dfactory.runtime import RuntimeContext
from dfactory.runtime.veomni import checkpoint, distributed, optimization, parallel


class VeOmniGateway:
    def initialize(self, config: DistributedConfig, *, seed: int) -> RuntimeContext:
        return distributed.initialize(config, seed=seed)

    def parallelize(
        self,
        model: Any,
        model_config: ModelConfig,
        distributed_config: DistributedConfig,
        *,
        sharding_plan: Any = None,
    ) -> Any:
        return parallel.parallelize(model, model_config, distributed_config, sharding_plan=sharding_plan)

    def activation_contexts(self, config: DistributedConfig) -> tuple[Any, Any]:
        return parallel.activation_contexts(config)

    def build_optimization(self, model: Any, config: OptimizationConfig, *, train_steps: int) -> tuple[Any, Any]:
        return optimization.build(model, config, train_steps=train_steps)

    def checkpoint_io(self, distributed_mode: str) -> checkpoint.CheckpointIO:
        return checkpoint.CheckpointIO(distributed_mode)

    def move_batch(self, batch: Any, device: str) -> Any:
        return distributed.move_to_device(batch, device)

    def reduce_loss(self, loss: Any, token_count: Any) -> Any:
        return distributed.reduce_loss(loss, token_count)

    def synchronize(self) -> None:
        distributed.synchronize()

    @staticmethod
    def clip_gradients(model: Any, max_norm: float) -> float:
        import torch

        if hasattr(model, "clip_grad_norm_"):
            value = model.clip_grad_norm_(max_norm)
        else:
            value = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        return float(value.item() if hasattr(value, "item") else value)


__all__ = ["VeOmniGateway"]
