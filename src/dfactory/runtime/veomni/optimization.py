"""Optimizer construction backed by VeOmni."""

from __future__ import annotations

from typing import Any

from veomni.optim import build_lr_scheduler, build_optimizer

from dfactory.config import OptimizationConfig


def build(model: Any, config: OptimizationConfig, *, train_steps: int) -> tuple[Any, Any]:
    optimizer = build_optimizer(
        model,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        fused=True,
        optimizer_type=config.optimizer,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        train_steps=train_steps,
        lr=config.lr,
        lr_min=config.min_lr,
        lr_decay_style=config.scheduler,
        lr_decay_ratio=config.decay_ratio,
        lr_warmup_ratio=config.warmup_ratio,
    )
    return optimizer, scheduler


__all__ = ["build"]
