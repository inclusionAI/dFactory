"""Canonical batch shape passed from data code to recipes."""

from typing import Any, NotRequired, TypedDict

import torch


class AttentionSpec(TypedDict):
    """Tensor metadata from which a recipe builds device-side attention."""

    sequence_lengths: torch.Tensor
    valid_lengths: torch.Tensor


class TrainBatch(TypedDict):
    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    attention_spec: AttentionSpec
    extras: NotRequired[dict[str, Any]]


__all__ = ["AttentionSpec", "TrainBatch"]
