"""Recipe metadata and construction contract shared by built-ins and plugins."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from dfactory.config import DataConfig, RunConfig
from dfactory.trainers.base import BaseTrainer


class MapDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


@dataclass(frozen=True)
class RecipeData:
    """Map-style training dataset and its optional batch collation function."""

    dataset: MapDataset
    collate_fn: Callable[[list[Any]], Any] | None = None


class RecipeRunValidator(Protocol):
    def __call__(self, *, recipe_config: Any, run_config: RunConfig) -> None: ...


class RecipeDataFactory(Protocol):
    def __call__(
        self,
        *,
        source: Any,
        tokenizer: Any,
        data_config: DataConfig,
        recipe_config: Any,
        special_tokens: Mapping[str, int],
    ) -> RecipeData: ...


@dataclass(frozen=True)
class RecipeSpec:
    name: str
    config_type: type[Any]
    validate_run: RecipeRunValidator
    data_factory: RecipeDataFactory
    trainer_type: type[BaseTrainer]
    description: str


__all__ = ["MapDataset", "RecipeData", "RecipeDataFactory", "RecipeRunValidator", "RecipeSpec"]
