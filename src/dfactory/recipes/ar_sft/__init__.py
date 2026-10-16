"""Built-in autoregressive supervised fine-tuning recipe."""

from dfactory.recipes.ar_sft.config import ARSFTConfig
from dfactory.recipes.ar_sft.data import build_data, build_transform, validate_run
from dfactory.recipes.ar_sft.trainer import ARSFTTrainer
from dfactory.recipes.spec import RecipeSpec

recipe_spec = RecipeSpec(
    name="ar_sft",
    config_type=ARSFTConfig,
    validate_run=validate_run,
    data_factory=build_data,
    trainer_type=ARSFTTrainer,
    description="Autoregressive supervised fine-tuning with document-isolated causal attention.",
)

__all__ = ["ARSFTConfig", "ARSFTTrainer", "build_data", "build_transform", "recipe_spec"]
