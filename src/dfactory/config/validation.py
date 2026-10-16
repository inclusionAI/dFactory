"""Validation rules for resolved dFactory configuration."""

from __future__ import annotations

from dfactory.config.schema import RunConfig


def validate_config(config: RunConfig) -> None:
    if not config.recipe.name:
        raise ValueError("recipe.name must not be empty")
    if not config.model.provider:
        raise ValueError("model.provider must not be empty")
    if not config.model.config_path:
        raise ValueError("model.config_path must not be empty")
    if not config.model.weights_path:
        raise ValueError("model.weights_path must not be empty")
    if not config.model.tokenizer_path:
        raise ValueError("model.tokenizer_path must not be empty")
    if config.model.attention not in ("eager", "sdpa", "flex_attention"):
        raise ValueError("model.attention must be one of: eager, sdpa, flex_attention")
    if config.model.moe not in ("fused", "eager"):
        raise ValueError("model.moe must be one of: fused, eager")
    if config.model.init_device != "meta":
        raise ValueError("model.init_device must be meta for FSDP2")
    if not config.data.source:
        raise ValueError("data.source must not be empty")
    if not config.data.path:
        raise ValueError("data.path must not be empty")
    if config.data.max_seq_len <= 0:
        raise ValueError("data.max_seq_len must be positive")
    if config.run.max_steps <= 0:
        raise ValueError("run.max_steps must be positive")
    if config.run.num_train_epochs <= 0:
        raise ValueError("run.num_train_epochs must be positive")
    if config.optimization.global_batch_size <= 0 or config.optimization.micro_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if config.optimization.max_grad_norm <= 0:
        raise ValueError("optimization.max_grad_norm must be positive")
    if config.optimization.optimizer not in ("adamw", "anyprecision_adamw"):
        raise ValueError("optimization.optimizer must be one of: adamw, anyprecision_adamw")
    if config.optimization.scheduler not in ("constant", "linear", "cosine"):
        raise ValueError("optimization.scheduler must be one of: constant, linear, cosine")
    if not 0.0 <= config.optimization.warmup_ratio <= 1.0:
        raise ValueError("optimization.warmup_ratio must be in [0, 1]")
    if config.logging.log_every_steps <= 0:
        raise ValueError("logging.log_every_steps must be positive")
    if not isinstance(config.logging.report_to, list) or any(target != "stdout" for target in config.logging.report_to):
        raise ValueError("logging.report_to may contain only: stdout")
    if config.checkpoint.save_every_steps is not None and config.checkpoint.save_every_steps <= 0:
        raise ValueError("checkpoint.save_every_steps must be positive")
    if config.checkpoint.save_every_epochs is not None and config.checkpoint.save_every_epochs <= 0:
        raise ValueError("checkpoint.save_every_epochs must be positive")
    if config.distributed.mode != "fsdp2":
        raise ValueError("distributed.mode must be fsdp2")
    if config.distributed.dp < 2:
        raise ValueError("distributed.dp must be at least 2; single-GPU training is not supported")
    for name in ("dp", "dp_replicate", "ep"):
        if getattr(config.distributed, name) <= 0:
            raise ValueError(f"distributed.{name} must be positive")
    if config.distributed.dp % config.distributed.dp_replicate:
        raise ValueError("distributed.dp must be divisible by distributed.dp_replicate")
    if config.distributed.dp % config.distributed.ep:
        raise ValueError("distributed.ep must divide distributed.dp")
    if config.model.moe == "eager" and config.distributed.ep > 1:
        raise ValueError("model.moe=eager does not support distributed.ep > 1")
    denominator = config.optimization.micro_batch_size * config.distributed.dp
    if config.optimization.global_batch_size % denominator:
        raise ValueError("optimization.global_batch_size must be divisible by micro_batch_size * distributed.dp")


__all__ = ["validate_config"]
