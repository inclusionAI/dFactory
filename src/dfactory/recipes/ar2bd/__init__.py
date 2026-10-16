"""Built-in autoregressive-to-Block-Diffusion pretraining recipe."""

from dfactory.recipes.ar2bd.config import AR2BDConfig
from dfactory.recipes.ar2bd.data import build_data, build_transform, validate_run
from dfactory.recipes.ar2bd.trainer import AR2BDTrainer
from dfactory.recipes.spec import RecipeSpec

recipe_spec = RecipeSpec(
    name="ar2bd",
    config_type=AR2BDConfig,
    validate_run=validate_run,
    data_factory=build_data,
    trainer_type=AR2BDTrainer,
    description="Continued pretraining from autoregressive tokens to Block Diffusion.",
)

__all__ = ["AR2BDConfig", "AR2BDTrainer", "build_data", "build_transform", "recipe_spec"]
