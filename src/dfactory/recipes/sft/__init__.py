"""Built-in LLaDA2 SFT recipe."""

from dfactory.recipes.sft.config import SFTConfig
from dfactory.recipes.sft.data import build_data, build_transform, validate_run
from dfactory.recipes.sft.trainer import SFTTrainer
from dfactory.recipes.spec import RecipeSpec

recipe_spec = RecipeSpec(
    name="sft",
    config_type=SFTConfig,
    validate_run=validate_run,
    data_factory=build_data,
    trainer_type=SFTTrainer,
    description="Block Diffusion supervised fine-tuning for LLaDA2.",
)

__all__ = ["SFTConfig", "SFTTrainer", "build_data", "build_transform", "recipe_spec"]
