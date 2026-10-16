"""Built-in EMTF recipe with one random self-correcting stage."""

from dfactory.recipes.emtf_random_t2t.config import EMTFRandomT2TConfig
from dfactory.recipes.emtf_random_t2t.data import build_data, build_transform, validate_run
from dfactory.recipes.emtf_random_t2t.trainer import EMTFRandomT2TTrainer
from dfactory.recipes.spec import RecipeSpec

recipe_spec = RecipeSpec(
    name="emtf_random_t2t",
    config_type=EMTFRandomT2TConfig,
    validate_run=validate_run,
    data_factory=build_data,
    trainer_type=EMTFRandomT2TTrainer,
    description="EMTF with teacher forcing at every stage and one random self-correcting stage.",
)

__all__ = [
    "EMTFRandomT2TConfig",
    "EMTFRandomT2TTrainer",
    "build_data",
    "build_transform",
    "recipe_spec",
]
