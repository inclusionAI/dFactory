"""Built-in multi-stage teacher-forcing recipe."""

from dfactory.recipes.mtf.config import MTFConfig
from dfactory.recipes.mtf.data import build_data, build_transform, validate_run
from dfactory.recipes.mtf.trainer import MTFTrainer
from dfactory.recipes.spec import RecipeSpec

recipe_spec = RecipeSpec(
    name="mtf",
    config_type=MTFConfig,
    validate_run=validate_run,
    data_factory=build_data,
    trainer_type=MTFTrainer,
    description="Multi-stage confidence curriculum with teacher forcing only.",
)

__all__ = ["MTFConfig", "MTFTrainer", "build_data", "build_transform", "recipe_spec"]
