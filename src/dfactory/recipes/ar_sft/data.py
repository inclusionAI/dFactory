"""Autoregressive SFT data transformation and packing assembly."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from dfactory.config import DataConfig, RunConfig
from dfactory.data import PackedDataset, TrainBatch
from dfactory.data.collators import collate_train_batch
from dfactory.data.transforms import TransformDataset, transform_conversation
from dfactory.recipes import RecipeData
from dfactory.recipes.ar_sft.config import ARSFTConfig


def build_transform(
    *,
    tokenizer: Any,
    data_config: DataConfig,
    recipe_config: ARSFTConfig,
    special_tokens: Mapping[str, int],
) -> Callable[[dict[str, Any]], TrainBatch]:
    del recipe_config
    if data_config.format != "conversation":
        raise ValueError("the AR SFT recipe supports data.format=conversation only")
    return partial(
        transform_conversation,
        tokenizer=tokenizer,
        max_seq_len=data_config.max_seq_len,
        block_size=1,
        pad_token_id=special_tokens["pad_token_id"],
        text_key=data_config.text_key,
        pad_to_max_length=not data_config.packing,
    )


def validate_run(*, recipe_config: ARSFTConfig, run_config: RunConfig) -> None:
    del recipe_config
    if run_config.data.format != "conversation":
        raise ValueError("the AR SFT recipe supports data.format=conversation only")


def build_data(
    *,
    source: Any,
    tokenizer: Any,
    data_config: DataConfig,
    recipe_config: ARSFTConfig,
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
