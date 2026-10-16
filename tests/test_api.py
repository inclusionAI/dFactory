from pathlib import Path

import pytest

from dfactory import RunConfig, train
from dfactory.config import DataConfig, DistributedConfig, ModelConfig


def _run_config(*, dp: int = 2) -> RunConfig:
    return RunConfig(
        model=ModelConfig(config_path="config", weights_path="weights", tokenizer_path="tokenizer"),
        data=DataConfig(path="data"),
        distributed=DistributedConfig(dp=dp),
    )


def test_train_validates_a_constructed_run_config() -> None:
    with pytest.raises(ValueError, match="distributed.dp must be at least 2"):
        train(_run_config(dp=1))


def test_train_rejects_overrides_for_a_constructed_run_config() -> None:
    with pytest.raises(ValueError, match="overrides can only be used when config is a YAML path"):
        train(_run_config(), overrides=["run.max_steps=2"])


def test_train_still_accepts_a_path_with_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = _run_config()
    observed: dict[str, object] = {}

    class FakeRecipeConfig:
        @classmethod
        def from_mapping(cls, values):
            assert values == {}
            return cls()

    class FakeTrainer:
        def __init__(self, *, run_config, recipe_config, data_factory):
            observed.update(
                config=run_config,
                recipe_config=recipe_config,
                data_factory=data_factory,
            )

        def train(self):
            observed["trained"] = True

    recipe_spec = type(
        "RecipeSpec",
        (),
        {
            "config_type": FakeRecipeConfig,
            "validate_run": staticmethod(lambda **kwargs: observed.update(validated=kwargs)),
            "data_factory": staticmethod(lambda **kwargs: None),
            "trainer_type": FakeTrainer,
        },
    )

    def fake_load_config(path, *, overrides):
        observed["path"] = path
        observed["overrides"] = overrides
        return expected

    def fake_get_recipe(name):
        observed["recipe_name"] = name
        return recipe_spec

    monkeypatch.setattr("dfactory.api.load_config", fake_load_config)
    monkeypatch.setattr("dfactory.recipes.get_recipe", fake_get_recipe)
    path = tmp_path / "run.yaml"

    train(path, overrides=["run.max_steps=2"])

    assert observed["path"] == path
    assert observed["overrides"] == ["run.max_steps=2"]
    assert observed["recipe_name"] == "sft"
    assert observed["config"] is expected
    assert isinstance(observed["recipe_config"], FakeRecipeConfig)
    assert observed["data_factory"] is recipe_spec.data_factory
    assert observed["validated"] == {
        "recipe_config": observed["recipe_config"],
        "run_config": expected,
    }
    assert observed["trained"] is True
