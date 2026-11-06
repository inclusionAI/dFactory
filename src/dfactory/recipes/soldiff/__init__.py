"""Block-internal SolDiff pretraining with a trainable q ordering model."""

from dfactory.recipes.soldiff.config import SolDiffConfig
from dfactory.recipes.soldiff.data import FixedTokenRows, build_data, validate_run
from dfactory.recipes.soldiff.trainer import SolDiffTrainer
from dfactory.recipes.spec import RecipeSpec

recipe_spec = RecipeSpec(
    name="soldiff",
    config_type=SolDiffConfig,
    validate_run=validate_run,
    data_factory=build_data,
    trainer_type=SolDiffTrainer,
    description="Qwen3 block-internal SolDiff with RLOO, prior permutation alignment, and suffix reconstruction.",
)

__all__ = ["FixedTokenRows", "SolDiffConfig", "SolDiffTrainer", "build_data", "recipe_spec"]
