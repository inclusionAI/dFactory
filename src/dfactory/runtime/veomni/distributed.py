"""Distributed operations delegated to VeOmni 0.1.4."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.sequence_parallel import reduce_sequence_parallel_loss
from veomni.utils.helper import set_seed

from dfactory.config import DistributedConfig
from dfactory.runtime import RuntimeContext


def initialize(config: DistributedConfig, *, seed: int) -> RuntimeContext:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("dFactory training requires x86-64 Linux")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"dFactory training requires a CUDA 12.8 PyTorch build, got {torch.version.cuda!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("dFactory training requires CUDA")
    if not dist.is_nccl_available():
        raise RuntimeError("dFactory distributed training requires NCCL")
    if config.dp < 2:
        raise ValueError("dFactory training requires distributed.dp >= 2; single-GPU training is not supported")

    required_environment = ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    missing_environment = [name for name in required_environment if name not in os.environ]
    if missing_environment:
        missing = ", ".join(missing_environment)
        raise RuntimeError(f"dFactory training must be launched with torchrun; missing environment: {missing}")

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    configured_world_size = int(os.environ["WORLD_SIZE"])
    if local_world_size < 2:
        raise ValueError("dFactory training requires at least two processes per node")
    if not 0 <= local_rank < local_world_size:
        raise ValueError(f"torchrun LOCAL_RANK ({local_rank}) must be in [0, LOCAL_WORLD_SIZE ({local_world_size}))")
    visible_devices = torch.cuda.device_count()
    if visible_devices < local_world_size:
        raise RuntimeError(
            f"torchrun LOCAL_WORLD_SIZE ({local_world_size}) exceeds visible CUDA devices ({visible_devices})"
        )
    if configured_world_size != config.dp:
        raise ValueError(f"torchrun WORLD_SIZE ({configured_world_size}) must equal distributed.dp ({config.dp})")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    elif dist.get_backend() != "nccl":
        raise RuntimeError(f"dFactory requires an NCCL process group, got {dist.get_backend()}")

    world_size = dist.get_world_size()
    if world_size != config.dp:
        raise ValueError(f"process-group world size ({world_size}) must equal distributed.dp ({config.dp})")
    set_seed(seed, full_determinism=False)

    if config.dp % config.dp_replicate:
        raise ValueError("distributed.dp must be divisible by distributed.dp_replicate")
    init_parallel_state(
        dp_size=config.dp,
        dp_replicate_size=config.dp_replicate,
        dp_shard_size=config.dp // config.dp_replicate,
        tp_size=1,
        ep_size=config.ep,
        ulysses_size=1,
        dp_mode=config.mode,
        device_type="cuda",
    )
    state = get_parallel_state()
    return RuntimeContext(
        device="cuda",
        global_rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=world_size,
        data_parallel_size=state.dp_size,
        data_parallel_rank=state.dp_rank,
        tensor_parallel_size=state.tp_size,
        expert_parallel_size=state.ep_size,
        sequence_parallel_size=state.ulysses_size,
    )


def move_to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def reduce_loss(loss: torch.Tensor, token_count: torch.Tensor) -> torch.Tensor:
    state = get_parallel_state()
    if state.sp_enabled:
        from veomni.distributed.sequence_parallel import get_unified_sequence_parallel_group

        total_token_count = token_count.detach().clone()
        dist.all_reduce(total_token_count, group=get_unified_sequence_parallel_group())
        if total_token_count.item() == 0:
            return loss * 0.0
        return reduce_sequence_parallel_loss(loss, token_count)
    return loss


def synchronize() -> None:
    torch.cuda.synchronize()


__all__ = ["initialize", "move_to_device", "reduce_loss", "synchronize"]
