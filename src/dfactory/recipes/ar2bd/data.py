"""Token-record transformation and packing for AR2BD pretraining."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from dfactory.config import DataConfig, RunConfig
from dfactory.data import PackedDataset, TrainBatch
from dfactory.data.collators import collate_train_batch
from dfactory.data.transforms.sft import TransformDataset
from dfactory.data.transforms.token_records import transform_token_record
from dfactory.recipes import RecipeData
from dfactory.recipes.ar2bd.config import AR2BDConfig


def build_transform(
    *,
    tokenizer: Any,
    data_config: DataConfig,
    recipe_config: AR2BDConfig,
    special_tokens: Mapping[str, int],
) -> Callable[[dict[str, Any]], TrainBatch]:
    del tokenizer
    if data_config.format != "token_records":
        raise ValueError("the AR2BD recipe supports data.format=token_records only")
    return partial(
        transform_token_record,
        max_seq_len=data_config.max_seq_len,
        block_size=recipe_config.block_size,
        pad_token_id=special_tokens["pad_token_id"],
        token_key=data_config.text_key,
        pad_to_max_length=not data_config.packing,
    )


def validate_run(*, recipe_config: AR2BDConfig, run_config: RunConfig) -> None:
    recipe_config.validate()
    if run_config.data.format != "token_records":
        raise ValueError("the AR2BD recipe supports data.format=token_records only")
    if run_config.data.max_seq_len % recipe_config.block_size:
        raise ValueError("data.max_seq_len must be divisible by recipe.params.block_size")


def build_data(
    *,
    source: Any,
    tokenizer: Any,
    data_config: DataConfig,
    recipe_config: AR2BDConfig,
    special_tokens: Mapping[str, int],
) -> RecipeData:
    transform = build_transform(
        tokenizer=tokenizer,
        data_config=data_config,
        recipe_config=recipe_config,
        special_tokens=special_tokens,
    )
    dataset = TransformDataset(source, transform)
    if data_config.packing:
        dataset = PackedDataset(
            dataset,
            max_seq_len=data_config.max_seq_len,
            pad_token_id=special_tokens["pad_token_id"],
        )
    return RecipeData(dataset=dataset, collate_fn=collate_train_batch)


__all__ = ["build_data", "build_transform", "validate_run"]
