import json
import tomllib
from pathlib import Path

import dfactory


def test_public_api_is_importable() -> None:
    assert dfactory.__version__ == "0.1.0"
    assert callable(dfactory.train)
    assert dfactory.AttentionSpec.__module__ == "dfactory.data.batch"
    assert dfactory.DatasetSpec.__module__ == "dfactory.data.spec"
    assert dfactory.ModelSpec.__module__ == "dfactory.models.spec"
    assert dfactory.RecipeData.__module__ == "dfactory.recipes.spec"
    assert dfactory.RunConfig.__module__ == "dfactory.config.schema"
    assert dfactory.BaseTrainer.__module__ == "dfactory.trainers.base"
    assert dfactory.StepContext.__module__ == "dfactory.trainers.context"
    assert dfactory.StepMetrics.__module__ == "dfactory.trainers.metrics"
    assert dfactory.TrainerState.__module__ == "dfactory.trainers.state"
    assert callable(dfactory.list_datasets)
    assert callable(dfactory.list_models)
    assert callable(dfactory.list_recipes)
    assert "_helpers" not in dfactory.__all__
    assert dfactory.TrainBatch.__required_keys__ == {
        "input_ids",
        "labels",
        "loss_mask",
        "position_ids",
        "attention_spec",
    }


def test_project_name_is_dfactory() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert metadata["project"]["name"] == "dfactory"
    assert metadata["project"]["license-files"] == ["LICENSE", "LEGAL.md"]


def test_builtin_recipe_packages_own_their_trainers() -> None:
    source_root = Path(__file__).parents[1] / "src" / "dfactory"
    recipe_names = {"ar2bd", "ar_sft", "emtf", "emtf_random_t2t", "estf", "mtf", "sft"}

    assert {path.parent.name for path in (source_root / "recipes").glob("*/trainer.py")} == recipe_names
    assert {path.name for path in (source_root / "trainers").glob("*.py")} == {
        "__init__.py",
        "base.py",
        "context.py",
        "metrics.py",
        "state.py",
    }


def test_cuda_runtime_is_the_default_supported_environment() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    dependencies = set(metadata["project"]["dependencies"])
    assert "huggingface-hub" in dependencies
    assert "safetensors" in dependencies
    assert "torch==2.8.0" in dependencies
    assert "torchaudio==2.8.0" in dependencies
    assert "torchvision==0.23.0" in dependencies
    assert "veomni[gpu]==0.1.4" in dependencies
    assert "gpu" not in metadata["project"]["optional-dependencies"]
    assert metadata["tool"]["uv"]["environments"] == ["sys_platform == 'linux' and platform_machine == 'x86_64'"]


def test_model_configs_do_not_duplicate_python_model_implementations() -> None:
    config_root = Path(__file__).parents[1] / "configs" / "model_configs"
    for config_path in config_root.glob("*/config.json"):
        assert "auto_map" not in json.loads(config_path.read_text(encoding="utf-8"))
        assert list(config_path.parent.glob("*.py")) == []
