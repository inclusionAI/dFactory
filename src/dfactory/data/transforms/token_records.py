"""Pre-tokenized document records for continued pretraining recipes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch

from dfactory.components.block_diffusion.layout import align_to_block
from dfactory.data.batch import TrainBatch

IGNORE_INDEX = -100


def _token_ids(record: Mapping[str, Any], token_key: str) -> list[int]:
    if token_key not in record:
        raise ValueError(f"token record is missing field {token_key!r}")
    value = record[token_key]
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.dtype == torch.bool or value.is_floating_point():
            raise ValueError(f"token record field {token_key!r} must be a one-dimensional integer sequence")
        tokens = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        tokens = list(value)
    else:
        raise ValueError(f"token record field {token_key!r} must be a one-dimensional integer sequence")
    if not tokens:
        raise ValueError(f"token record field {token_key!r} must not be empty")
    if any(not isinstance(token, Integral) or isinstance(token, bool) for token in tokens):
        raise ValueError(f"token record field {token_key!r} must contain integers only")
    return [int(token) for token in tokens]


def transform_token_record(
    record: Mapping[str, Any],
    *,
    max_seq_len: int,
    block_size: int,
    pad_token_id: int,
    token_key: str = "input_ids",
    pad_to_max_length: bool = True,
) -> TrainBatch:
    """Convert one pre-tokenized document without sampling training noise.

    A record represents one document. Its block-aligned capacity becomes the
    document boundary consumed by Block Diffusion packing, while
    ``valid_lengths`` keeps padding out of attention, noise, and the objective.
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if block_size <= 0 or max_seq_len % block_size:
        raise ValueError("max_seq_len must be divisible by a positive block_size")

    tokens = _token_ids(record, token_key)[:max_seq_len]
    valid_length = len(tokens)
    target_length = max_seq_len if pad_to_max_length else align_to_block(valid_length, block_size)
    padding = target_length - valid_length

    input_ids = torch.tensor(tokens + [pad_token_id] * padding, dtype=torch.long)
    loss_mask = torch.arange(target_length) < valid_length
    labels = input_ids.masked_fill(~loss_mask, IGNORE_INDEX)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": torch.arange(target_length, dtype=torch.long),
        "attention_spec": {
            "sequence_lengths": torch.tensor([target_length], dtype=torch.long),
            "valid_lengths": torch.tensor([valid_length], dtype=torch.long),
        },
    }


__all__ = ["IGNORE_INDEX", "transform_token_record"]
