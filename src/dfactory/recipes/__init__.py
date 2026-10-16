"""Training recipe contracts and discovery."""

from __future__ import annotations

from dfactory.recipes.spec import RecipeData, RecipeSpec
from dfactory.registry import Registry

_RECIPES: Registry[RecipeSpec] = Registry("dfactory.recipes")


def register_recipe(spec: RecipeSpec) -> None:
    _register_builtins()
    _RECIPES.register(spec.name, spec)


def get_recipe(name: str) -> RecipeSpec:
    _register_builtins()
    return _RECIPES.get(name)


def list_recipes() -> tuple[str, ...]:
    _register_builtins()
    return _RECIPES.names()


def describe_recipe(name: str) -> RecipeSpec:
    return get_recipe(name)


def _register_builtins() -> None:
    builtins = {
        "ar2bd": ("dfactory.recipes.ar2bd", "recipe_spec"),
        "ar_sft": ("dfactory.recipes.ar_sft", "recipe_spec"),
        "emtf": ("dfactory.recipes.emtf", "recipe_spec"),
        "emtf_random_t2t": ("dfactory.recipes.emtf_random_t2t", "recipe_spec"),
        "estf": ("dfactory.recipes.estf", "recipe_spec"),
        "mtf": ("dfactory.recipes.mtf", "recipe_spec"),
        "sft": ("dfactory.recipes.sft", "recipe_spec"),
    }
    for name, (module, attribute) in builtins.items():
        if not _RECIPES.contains_builtin(name):
            _RECIPES.register_lazy(name, module, attribute)


__all__ = [
    "RecipeData",
    "RecipeSpec",
    "describe_recipe",
    "get_recipe",
    "list_recipes",
    "register_recipe",
]
