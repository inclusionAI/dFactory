"""Stable public entry points for dFactory applications and plugins."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

from dfactory.config import RunConfig, load_config
from dfactory.config.validation import validate_config


def train(config: RunConfig | str | PathLike[str], overrides: Sequence[str] = ()) -> None:
    """Run training from a resolved config or a YAML config path."""
    if isinstance(config, RunConfig):
        if overrides:
            raise ValueError("overrides can only be used when config is a YAML path")
        validate_config(config)
    else:
        config = load_config(config, overrides=overrides)

    from dfactory.recipes import get_recipe

    recipe_spec = get_recipe(config.recipe.name)
    recipe_config = recipe_spec.config_type.from_mapping(config.recipe.params)
    recipe_spec.validate_run(recipe_config=recipe_config, run_config=config)
    trainer = recipe_spec.trainer_type(
        run_config=config,
        recipe_config=recipe_config,
        data_factory=recipe_spec.data_factory,
    )
    trainer.train()


__all__ = ["RunConfig", "train"]
