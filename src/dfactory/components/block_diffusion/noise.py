"""Runtime noise primitives for Block Diffusion recipes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

IGNORE_INDEX = -100


@dataclass(frozen=True)
class NoiseResult:
    input_ids: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor


def apply_block_noise(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    trainable_mask: torch.Tensor,
    noisy_slot_mask: torch.Tensor,
    noise_group_ids: torch.Tensor,
    mask_token_id: int,
    eos_token_id: int | None = None,
    noise_range: tuple[float, float],
    force_mask_token_ids: Sequence[int] = (),
    generator: torch.Generator | None = None,
) -> NoiseResult:
    """Mask trainable noisy-slot tokens with one sampled rate per non-negative group."""
    tensors = (labels, trainable_mask, noisy_slot_mask, noise_group_ids)
    if any(tensor.shape != input_ids.shape for tensor in tensors):
        raise ValueError("all noise inputs must have the same shape")
    low, high = noise_range
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("noise_range must satisfy 0 <= low <= high <= 1")

    trainable = trainable_mask.bool()
    noisy_slots = noisy_slot_mask.bool()
    eligible = trainable & noisy_slots
    noised_source = input_ids
    effective_labels = labels
    if eos_token_id is not None:
        force_eos = torch.zeros_like(eligible)
        for batch_index in range(input_ids.shape[0]):
            groups = torch.unique(noise_group_ids[batch_index][noisy_slots[batch_index]])
            for group in groups[groups >= 0]:
                positions = torch.nonzero(
                    (noise_group_ids[batch_index] == group) & noisy_slots[batch_index],
                    as_tuple=False,
                ).flatten()
                group_trainable = trainable[batch_index, positions]
                transitions = torch.nonzero(group_trainable[:-1] & ~group_trainable[1:], as_tuple=False)
                if transitions.numel():
                    force_eos[batch_index, positions[int(transitions[0].item()) + 1 :]] = True
        if torch.any(force_eos):
            noised_source = input_ids.clone()
            noised_source[force_eos] = eos_token_id
            effective_labels = labels.clone()
            effective_labels[force_eos] = eos_token_id
            eligible = eligible | force_eos

    probabilities = torch.zeros(input_ids.shape, dtype=torch.float32, device=input_ids.device)
    for batch_index in range(input_ids.shape[0]):
        groups = torch.unique(noise_group_ids[batch_index][eligible[batch_index]])
        groups = groups[groups >= 0]
        if not groups.numel():
            continue
        samples = torch.rand(groups.numel(), device=input_ids.device, generator=generator)
        samples = samples * (high - low) + low
        for group, probability in zip(groups, samples, strict=True):
            probabilities[batch_index][noise_group_ids[batch_index] == group] = probability

    random_values = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    mask = eligible & (random_values < probabilities)
    for token_id in force_mask_token_ids:
        mask |= eligible & (noised_source == token_id)

    noised = noised_source.clone()
    noised[mask] = mask_token_id
    updated_labels = effective_labels.clone()
    updated_labels[eligible & ~mask] = IGNORE_INDEX
    return NoiseResult(input_ids=noised, labels=updated_labels, mask=mask)


__all__ = ["IGNORE_INDEX", "NoiseResult", "apply_block_noise"]
