"""Noisy/clean pair layout shared by Block Diffusion recipes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockLayout:
    pair_ids: torch.Tensor
    slot_types: torch.Tensor
    local_block_ids: torch.Tensor
    noise_group_ids: torch.Tensor
    valid_mask: torch.Tensor
    noisy_slot_mask: torch.Tensor
    position_ids: torch.Tensor


def interleave_paired_slots(
    values: torch.Tensor,
    sequence_lengths: torch.Tensor | Sequence[Sequence[int]],
    *,
    clean_values: torch.Tensor | None = None,
    fill_value: int | bool = 0,
) -> torch.Tensor:
    """Lay out packed values as ``[noisy_1, clean_1, noisy_2, clean_2, ...]``."""
    lengths = torch.as_tensor(sequence_lengths, dtype=torch.long, device=values.device)
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(0)
    if values.ndim != 2 or lengths.ndim != 2 or values.shape[0] != lengths.shape[0]:
        raise ValueError("values and sequence_lengths must have compatible batch dimensions")
    if torch.any(lengths < 0) or torch.any(lengths.sum(dim=1) > values.shape[1]):
        raise ValueError("sequence_lengths must describe slices within values")

    clean = values if clean_values is None else clean_values
    if clean.shape != values.shape:
        raise ValueError("clean_values must have the same shape as values")
    paired = torch.full(
        (values.shape[0], values.shape[1] * 2),
        fill_value,
        dtype=values.dtype,
        device=values.device,
    )
    for batch_index, sample_lengths in enumerate(lengths.tolist()):
        source_offset = 0
        target_offset = 0
        for length in sample_lengths:
            if length == 0:
                continue
            source_slice = slice(source_offset, source_offset + length)
            paired[batch_index, target_offset : target_offset + length] = values[batch_index, source_slice]
            target_offset += length
            paired[batch_index, target_offset : target_offset + length] = clean[batch_index, source_slice]
            source_offset += length
            target_offset += length
    return paired


def align_to_block(length: int, block_size: int) -> int:
    if length < 0:
        raise ValueError("length must not be negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return ((length + block_size - 1) // block_size) * block_size


def build_paired_layout(
    sequence_lengths: torch.Tensor | Sequence[Sequence[int]],
    *,
    block_size: int,
    total_length: int | None = None,
    valid_lengths: torch.Tensor | Sequence[Sequence[int]] | None = None,
) -> BlockLayout:
    """Describe packed ``[noisy, clean]`` pairs at token granularity."""
    lengths = torch.as_tensor(sequence_lengths, dtype=torch.long)
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(0)
    if lengths.ndim != 2:
        raise ValueError("sequence_lengths must have shape [batch, pairs]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if torch.any(lengths < 0):
        raise ValueError("sequence lengths must not be negative")
    nonzero = lengths[lengths > 0]
    if nonzero.numel() and torch.any(nonzero % block_size):
        raise ValueError("non-zero sequence lengths must be block aligned")

    valid = (
        lengths if valid_lengths is None else torch.as_tensor(valid_lengths, dtype=torch.long, device=lengths.device)
    )
    if valid.shape != lengths.shape:
        raise ValueError("valid_lengths must have the same shape as sequence_lengths")
    if torch.any(valid < 0) or torch.any(valid > lengths):
        raise ValueError("valid_lengths must satisfy 0 <= valid length <= sequence length")

    required_lengths = lengths.sum(dim=1) * 2
    inferred_length = int(required_lengths.max().item()) if required_lengths.numel() else 0
    total_length = inferred_length if total_length is None else total_length
    if total_length < inferred_length:
        raise ValueError(f"total_length={total_length} is smaller than required length {inferred_length}")

    batch_size = lengths.shape[0]
    device = lengths.device
    pair_ids = torch.full((batch_size, total_length), -1, dtype=torch.long, device=device)
    slot_types = torch.full_like(pair_ids, -1)
    local_block_ids = torch.full_like(pair_ids, -1)
    noise_group_ids = torch.full_like(pair_ids, -1)
    valid_mask = torch.zeros((batch_size, total_length), dtype=torch.bool, device=device)
    noisy_slot_mask = torch.zeros_like(valid_mask)
    position_ids = torch.zeros((batch_size, total_length), dtype=torch.long, device=device)

    for batch_index, sample_lengths in enumerate(lengths.tolist()):
        offset = 0
        group_offset = 0
        for pair_index, length in enumerate(sample_lengths):
            if length == 0:
                continue
            local_positions = torch.arange(length, device=device)
            local_blocks = local_positions // block_size
            blocks_per_pair = length // block_size
            valid_length = int(valid[batch_index, pair_index].item())
            for slot_type in (0, 1):
                end = offset + length
                token_slice = slice(offset, end)
                pair_ids[batch_index, token_slice] = pair_index
                slot_types[batch_index, token_slice] = slot_type
                local_block_ids[batch_index, token_slice] = local_blocks
                noise_group_ids[batch_index, token_slice] = group_offset + local_blocks
                valid_mask[batch_index, offset : offset + valid_length] = True
                noisy_slot_mask[batch_index, token_slice] = slot_type == 0
                position_ids[batch_index, token_slice] = local_positions
                offset = end
            group_offset += blocks_per_pair

    return BlockLayout(
        pair_ids=pair_ids,
        slot_types=slot_types,
        local_block_ids=local_block_ids,
        noise_group_ids=noise_group_ids,
        valid_mask=valid_mask,
        noisy_slot_mask=noisy_slot_mask,
        position_ids=position_ids,
    )


__all__ = ["BlockLayout", "align_to_block", "build_paired_layout", "interleave_paired_slots"]
