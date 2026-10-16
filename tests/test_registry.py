import pytest

import dfactory.data
import dfactory.models
import dfactory.recipes
from dfactory.registry import DuplicateProviderError, Registry


@pytest.mark.parametrize(
    ("module", "registry_name", "builtin_name", "getter"),
    [
        (dfactory.recipes, "_RECIPES", "sft", dfactory.recipes.get_recipe),
        (dfactory.models, "_MODELS", "llada2", dfactory.models.get_model),
        (dfactory.data, "_DATASETS", "jsonl", dfactory.data.get_dataset),
    ],
)
def test_external_provider_cannot_silently_replace_builtin(
    monkeypatch, module, registry_name: str, builtin_name: str, getter
) -> None:
    registry = Registry(f"test.{builtin_name}")
    monkeypatch.setattr(registry, "_external", lambda: {builtin_name: object()})
    monkeypatch.setattr(module, registry_name, registry)

    with pytest.raises(DuplicateProviderError, match="already registered"):
        getter(builtin_name)


def test_programmatic_recipe_cannot_replace_builtin(monkeypatch) -> None:
    registry = Registry("test.recipes")
    monkeypatch.setattr(dfactory.recipes, "_RECIPES", registry)

    with pytest.raises(DuplicateProviderError, match="already registered"):
        dfactory.recipes.register_recipe(type("RecipeSpec", (), {"name": "sft"})())
