from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import dfactory.runtime.veomni.distributed as distributed_module
from dfactory.config import DistributedConfig


def _set_torchrun_environment(
    monkeypatch, *, world_size: int = 2, local_world_size: int = 2, local_rank: int = 1
) -> None:
    values = {
        "MASTER_ADDR": "trainer-0",
        "MASTER_PORT": "29500",
        "RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "LOCAL_RANK": str(local_rank),
        "LOCAL_WORLD_SIZE": str(local_world_size),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _mock_cuda_contract(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "is_nccl_available", lambda: True)


def test_initialize_rejects_a_non_linux_host(monkeypatch) -> None:
    monkeypatch.setattr(distributed_module.sys, "platform", "darwin")

    with pytest.raises(RuntimeError, match="requires x86-64 Linux"):
        distributed_module.initialize(DistributedConfig(dp=2), seed=42)


def test_initialize_rejects_a_non_x86_64_host(monkeypatch) -> None:
    monkeypatch.setattr(distributed_module.platform, "machine", lambda: "aarch64")

    with pytest.raises(RuntimeError, match="requires x86-64 Linux"):
        distributed_module.initialize(DistributedConfig(dp=2), seed=42)


def test_initialize_rejects_a_non_cuda_12_8_build(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", "12.6")

    with pytest.raises(RuntimeError, match="requires a CUDA 12.8 PyTorch build"):
        distributed_module.initialize(DistributedConfig(dp=2), seed=42)


def test_initialize_rejects_a_host_without_cuda(monkeypatch) -> None:
    init_process_group = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(distributed_module.dist, "init_process_group", init_process_group)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        distributed_module.initialize(DistributedConfig(dp=2), seed=42)

    init_process_group.assert_not_called()


def test_initialize_rejects_a_torch_build_without_nccl(monkeypatch) -> None:
    init_process_group = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "is_nccl_available", lambda: False)
    monkeypatch.setattr(distributed_module.dist, "init_process_group", init_process_group)

    with pytest.raises(RuntimeError, match="requires NCCL"):
        distributed_module.initialize(DistributedConfig(dp=2), seed=42)

    init_process_group.assert_not_called()


def test_initialize_rejects_single_gpu_training(monkeypatch) -> None:
    _mock_cuda_contract(monkeypatch)

    with pytest.raises(ValueError, match="distributed.dp >= 2"):
        distributed_module.initialize(DistributedConfig(dp=1), seed=42)


def test_initialize_requires_torchrun_environment(monkeypatch) -> None:
    _mock_cuda_contract(monkeypatch)
    for name in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="launched with torchrun") as error:
        distributed_module.initialize(DistributedConfig(dp=2), seed=42)

    assert "MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE" in str(error.value)


def test_initialize_rejects_one_process_per_node(monkeypatch) -> None:
    _mock_cuda_contract(monkeypatch)
    _set_torchrun_environment(monkeypatch, local_world_size=1, local_rank=0)

    with pytest.raises(ValueError, match="at least two processes per node"):
        distributed_module.initialize(DistributedConfig(dp=2), seed=42)


def test_initialize_rejects_too_few_visible_devices(monkeypatch) -> None:
    _mock_cuda_contract(monkeypatch)
    _set_torchrun_environment(monkeypatch)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="exceeds visible CUDA devices"):
        distributed_module.initialize(DistributedConfig(dp=2), seed=42)


def test_initialize_rejects_world_size_that_differs_from_data_parallel_size(monkeypatch) -> None:
    _mock_cuda_contract(monkeypatch)
    _set_torchrun_environment(monkeypatch, world_size=4)
    init_process_group = Mock()
    monkeypatch.setattr(distributed_module.dist, "init_process_group", init_process_group)

    with pytest.raises(ValueError, match=r"WORLD_SIZE \(4\) must equal distributed.dp \(2\)"):
        distributed_module.initialize(DistributedConfig(dp=2), seed=42)

    init_process_group.assert_not_called()


def test_initialize_uses_cuda_and_nccl(monkeypatch) -> None:
    _mock_cuda_contract(monkeypatch)
    _set_torchrun_environment(monkeypatch)
    set_device = Mock()
    init_process_group = Mock()
    init_parallel_state = Mock()
    set_seed = Mock()
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed_module.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(distributed_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(distributed_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(distributed_module, "init_parallel_state", init_parallel_state)
    monkeypatch.setattr(
        distributed_module,
        "get_parallel_state",
        lambda: SimpleNamespace(dp_size=2, dp_rank=1, tp_size=1, ep_size=1, ulysses_size=1),
    )
    monkeypatch.setattr(distributed_module, "set_seed", set_seed)

    context = distributed_module.initialize(DistributedConfig(dp=2), seed=42)

    set_device.assert_called_once_with(1)
    init_process_group.assert_called_once_with(backend="nccl", device_id=torch.device("cuda", 1))
    init_parallel_state.assert_called_once_with(
        dp_size=2,
        dp_replicate_size=1,
        dp_shard_size=2,
        tp_size=1,
        ep_size=1,
        ulysses_size=1,
        dp_mode="fsdp2",
        device_type="cuda",
    )
    set_seed.assert_called_once_with(42, full_determinism=False)
    assert context.device == "cuda"
    assert context.world_size == 2
    assert context.local_rank == 1


def test_initialize_rejects_an_existing_non_nccl_process_group(monkeypatch) -> None:
    _mock_cuda_contract(monkeypatch)
    _set_torchrun_environment(monkeypatch)
    monkeypatch.setattr(torch.cuda, "set_device", Mock())
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "get_backend", lambda: "gloo")

    with pytest.raises(RuntimeError, match="requires an NCCL process group"):
        distributed_module.initialize(DistributedConfig(dp=2), seed=42)


def test_initialize_rejects_process_group_world_size_mismatch(monkeypatch) -> None:
    _mock_cuda_contract(monkeypatch)
    _set_torchrun_environment(monkeypatch)
    init_parallel_state = Mock()
    monkeypatch.setattr(torch.cuda, "set_device", Mock())
    monkeypatch.setattr(distributed_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_module.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(distributed_module.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(distributed_module, "init_parallel_state", init_parallel_state)

    with pytest.raises(ValueError, match=r"process-group world size \(4\) must equal distributed.dp \(2\)"):
        distributed_module.initialize(DistributedConfig(dp=2), seed=42)

    init_parallel_state.assert_not_called()
