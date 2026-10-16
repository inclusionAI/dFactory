"""Runtime topology visible to dFactory recipes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeContext:
    device: str
    global_rank: int
    local_rank: int
    world_size: int
    data_parallel_size: int
    data_parallel_rank: int
    tensor_parallel_size: int
    expert_parallel_size: int
    sequence_parallel_size: int


__all__ = ["RuntimeContext"]
