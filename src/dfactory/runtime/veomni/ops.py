"""Narrow exports of VeOmni primitives required by dFactory models."""

from __future__ import annotations

from typing import Any

from veomni.distributed.parallel_plan import ParallelPlan


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
    "fused_moe_forward",
    "get_parallel_state",
    "is_liger_kernel_available",
]
