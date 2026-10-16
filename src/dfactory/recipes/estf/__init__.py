"""Built-in ESTF continued-pretraining recipe."""

from dfactory.recipes.estf.algorithm import ESTFSelection, m2t_loss_weights, select_estf_tokens, token_confidence
from dfactory.recipes.estf.config import ESTFConfig
from dfactory.recipes.estf.data import build_data, build_transform, validate_run
from dfactory.recipes.estf.trainer import ESTFTrainer
from dfactory.recipes.spec import RecipeSpec

recipe_spec = RecipeSpec(
    name="estf",
    config_type=ESTFConfig,
    validate_run=validate_run,
    data_factory=build_data,
    trainer_type=ESTFTrainer,
    description="ESTF continued pretraining with paired masked-to-token and token-to-token passes.",
)

__all__ = [
    "ESTFConfig",
    "ESTFTrainer",
    "ESTFSelection",
    "build_data",
    "build_transform",
    "m2t_loss_weights",
    "recipe_spec",
    "select_estf_tokens",
    "token_confidence",
]
