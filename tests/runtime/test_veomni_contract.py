from importlib.metadata import version
from unittest.mock import Mock

from veomni.optim import build_lr_scheduler, build_optimizer

import dfactory.runtime.veomni.parallel as parallel_module
from dfactory.config import DistributedConfig, ModelConfig
from dfactory.runtime.veomni import VeOmniGateway


def test_supported_veomni_contract_is_available() -> None:
    assert version("veomni") == "0.1.4"
    assert callable(build_optimizer)
    assert callable(build_lr_scheduler)
    assert VeOmniGateway.__name__ == "VeOmniGateway"


def test_provider_parallel_policy_is_exposed_through_the_model(monkeypatch) -> None:
    model = type("ExternalModel", (), {"_no_split_modules": []})()
    policy = object()
    build_parallelize_model = Mock(return_value=model)
    monkeypatch.setattr(parallel_module, "build_parallelize_model", build_parallelize_model)

    result = parallel_module.parallelize(
        model,
        ModelConfig(weights_path="weights"),
        DistributedConfig(dp=2, ep=2),
        sharding_plan=policy,
    )

    assert result is model
    assert model.get_parallel_plan() is policy
    assert "sharding_plan" not in build_parallelize_model.call_args.kwargs
