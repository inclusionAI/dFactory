"""Installable causal language-model recipe example."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch

from dfactory import BaseTrainer, RecipeData, RecipeSpec, StepContext, StepMetrics, TrainBatch
from dfactory.components.block_diffusion import shifted_cross_entropy


@dataclass(frozen=True)
class ExampleRecipeConfig:
    loss_weight: float = 1.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ExampleRecipeConfig:
        unknown = sorted(set(values) - {item.name for item in fields(cls)})
        if unknown:
            raise ValueError(f"unknown example recipe parameter(s): {', '.join(unknown)}")
        config = cls(**values)
        if config.loss_weight <= 0:
            raise ValueError("recipe.params.loss_weight must be positive")
        return config


class TokenRecordDataset(Sequence[TrainBatch]):
    def __init__(self, source: Sequence[Mapping[str, Any]], *, max_seq_len: int, pad_token_id: int) -> None:
        self.source = source
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> TrainBatch:
        record = self.source[index]
        token_ids = list(record["input_ids"])[: self.max_seq_len]
        if len(token_ids) < 2:
            raise ValueError("example recipe records require at least two input_ids")
        pad_length = self.max_seq_len - len(token_ids)
        input_ids = torch.tensor(token_ids + [self.pad_token_id] * pad_length, dtype=torch.long)
        labels = input_ids.clone()
        labels[len(token_ids) :] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": labels != -100,
            "position_ids": torch.arange(self.max_seq_len, dtype=torch.long),
            "attention_spec": {
                "sequence_lengths": torch.tensor([self.max_seq_len], dtype=torch.long),
                "valid_lengths": torch.tensor([len(token_ids)], dtype=torch.long),
            },
        }


class ExampleTrainer(BaseTrainer):
    def training_step(self, batch: TrainBatch, context: StepContext) -> StepMetrics:
        outputs = context.forward({"input_ids": batch["input_ids"]})
        objective = shifted_cross_entropy(outputs.logits, batch["labels"])
        loss = context.reduce_loss(objective.loss, objective.token_count)
        context.backward(loss, weight=self.recipe_config.loss_weight)
        token_count = float(objective.token_count.detach().item())
        return StepMetrics(
            values={"loss": float(loss.detach().item()), "loss_tokens": token_count},
            weight=max(token_count, 1.0),
        )


def validate_run(*, recipe_config: ExampleRecipeConfig, run_config: Any) -> None:
    del recipe_config
    if run_config.data.format != "token_records":
        raise ValueError("example_causal_lm requires data.format=token_records")


def build_data(
    *,
    source: Any,
    tokenizer: Any,
    data_config: Any,
    recipe_config: ExampleRecipeConfig,
    special_tokens: Mapping[str, int],
) -> RecipeData:
    del tokenizer, recipe_config
    dataset = TokenRecordDataset(
        source,
        max_seq_len=data_config.max_seq_len,
        pad_token_id=special_tokens.get("pad_token_id", 0),
    )
    return RecipeData(dataset=dataset)


recipe_spec = RecipeSpec(
    name="example_causal_lm",
    config_type=ExampleRecipeConfig,
    validate_run=validate_run,
    data_factory=build_data,
    trainer_type=ExampleTrainer,
    description="Minimal causal language-model recipe implemented as an external plugin.",
)

__all__ = ["ExampleRecipeConfig", "ExampleTrainer", "TokenRecordDataset", "recipe_spec"]
