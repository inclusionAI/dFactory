"""Built-in editing multi-stage teacher-forcing recipe."""

from dfactory.recipes.emtf.config import EMTFConfig
from dfactory.recipes.emtf.data import build_data, build_transform, validate_run
from dfactory.recipes.emtf.trainer import EMTFTrainer
from dfactory.recipes.spec import RecipeSpec

recipe_spec = RecipeSpec(
    name="emtf",
    config_type=EMTFConfig,
    validate_run=validate_run,
    data_factory=build_data,
    trainer_type=EMTFTrainer,
    description="Multi-stage teacher forcing with a self-correcting phase.",
)

__all__ = ["EMTFConfig", "EMTFTrainer", "build_data", "build_transform", "recipe_spec"]
