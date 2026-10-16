"""Loss functions shared by Block Diffusion recipes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

IGNORE_INDEX = -100


@dataclass(frozen=True)
class ObjectiveResult:
    loss: torch.Tensor
    token_count: torch.Tensor


def same_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> ObjectiveResult:
    mask = labels != ignore_index
    token_count = mask.sum()
    if not torch.any(mask):
        return ObjectiveResult(loss=logits.sum() * 0.0, token_count=token_count)
    loss = functional.cross_entropy(logits[mask], labels[mask])
    return ObjectiveResult(loss=loss, token_count=token_count)


def row_mean_same_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> ObjectiveResult:
    """Average valid-token loss per row, then average the non-empty rows."""
    mask = labels != ignore_index
    token_count = mask.sum()
    if not torch.any(mask):
        return ObjectiveResult(loss=logits.sum() * 0.0, token_count=token_count)
    per_token = functional.cross_entropy(
        logits.transpose(1, 2),
        labels,
        reduction="none",
        ignore_index=ignore_index,
    )
    tokens_per_row = mask.sum(dim=-1)
    valid_rows = tokens_per_row > 0
    row_losses = (per_token * mask).sum(dim=-1)[valid_rows] / tokens_per_row[valid_rows]
    loss = row_losses.mean()
    return ObjectiveResult(loss=loss, token_count=token_count)


def shifted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> ObjectiveResult:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels != ignore_index
    token_count = mask.sum()
    if not torch.any(mask):
        return ObjectiveResult(loss=shifted_logits.sum() * 0.0, token_count=token_count)
    loss = functional.cross_entropy(shifted_logits[mask], shifted_labels[mask])
    return ObjectiveResult(loss=loss, token_count=token_count)


__all__ = [
    "IGNORE_INDEX",
    "ObjectiveResult",
    "row_mean_same_token_cross_entropy",
    "same_token_cross_entropy",
    "shifted_cross_entropy",
]
