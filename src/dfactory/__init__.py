"""dFactory public Python API."""

from dfactory.api import RunConfig, train
from dfactory.data import AttentionSpec, DatasetSpec, TrainBatch, get_dataset, list_datasets
from dfactory.models import ModelProvider, ModelSpec, get_model, list_models
from dfactory.recipes import (
    RecipeData,
    RecipeSpec,
    describe_recipe,
    get_recipe,
    list_recipes,
    register_recipe,
)
from dfactory.trainers import BaseTrainer, StepContext, StepMetrics, TrainerState

__all__ = [
    "AttentionSpec",
    "BaseTrainer",
    "DatasetSpec",
    "ModelProvider",
    "ModelSpec",
    "RecipeData",
    "RecipeSpec",
    "RunConfig",
    "StepContext",
    "StepMetrics",
    "TrainBatch",
    "TrainerState",
    "describe_recipe",
    "get_dataset",
    "get_model",
    "get_recipe",
    "list_datasets",
    "list_models",
    "list_recipes",
    "register_recipe",
    "train",
]

__version__ = "0.2.0.dev0"
