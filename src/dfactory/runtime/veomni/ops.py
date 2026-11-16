"""Narrow exports of VeOmni primitives required by dFactory models."""

from __future__ import annotations

from typing import Any

from veomni.distributed.parallel_plan import ParallelPlan
from veomni.models.module_utils import GradientCheckpointingLayer, init_empty_weights


def get_logger(name: str) -> Any:
    from veomni.utils.logging import get_logger as build_logger

    return build_logger(name)


def slice_position_embedding(*args: Any, **kwargs: Any) -> Any:
    from veomni.distributed.sequence_parallel import slice_position_embedding as slice_embedding

    return slice_embedding(*args, **kwargs)


def causallm_loss_function(
    hidden_states: Any,
    weight: Any,
    labels: Any,
    vocab_size: int | None = None,
    num_items_in_batch: int | None = None,
    ignore_index: int = -100,
    shift_labels: Any = None,
    **kwargs: Any,
) -> Any:
    # This is exactly VeOmni 0.1.4's labels=None branch. Keep dense
    # score-only forwards independent of its eager MoE/Triton package imports.
    assert shift_labels is None
    if labels is None:
        from torch.nn.functional import linear

        return None, linear(hidden_states, weight)
    from veomni.ops.loss import causallm_loss_function as compute_loss

    return compute_loss(
        hidden_states, weight, labels, vocab_size, num_items_in_batch, ignore_index, shift_labels, **kwargs
    )


def get_parallel_state() -> Any:
    from veomni.distributed.parallel_state import get_parallel_state as get_state

    return get_state()


def fused_moe_forward(*args: Any, **kwargs: Any) -> Any:
    from veomni.ops.fused_moe import fused_moe_forward as run_fused_moe

    return run_fused_moe(*args, **kwargs)


def is_liger_kernel_available() -> bool:
    from veomni.utils.import_utils import is_liger_kernel_available as is_available

    return is_available()


__all__ = [
    "ParallelPlan",
    "GradientCheckpointingLayer",
    "causallm_loss_function",
    "fused_moe_forward",
    "get_parallel_state",
    "get_logger",
    "is_liger_kernel_available",
    "init_empty_weights",
    "slice_position_embedding",
]
