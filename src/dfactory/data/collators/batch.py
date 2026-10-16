"""Collation for the public training-batch contract."""

from __future__ import annotations

from typing import Any

import torch

from dfactory.data.batch import TrainBatch


def collate_train_batch(examples: list[dict[str, Any]]) -> TrainBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    return {
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
        "labels": torch.stack([example["labels"] for example in examples]),
        "loss_mask": torch.stack([example["loss_mask"] for example in examples]),
        "position_ids": torch.stack([example["position_ids"] for example in examples]),
        "attention_spec": {
            "sequence_lengths": torch.stack([example["attention_spec"]["sequence_lengths"] for example in examples]),
            "valid_lengths": torch.stack([example["attention_spec"]["valid_lengths"] for example in examples]),
        },
    }


__all__ = ["collate_train_batch"]
