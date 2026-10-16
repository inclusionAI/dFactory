"""Training-side noise reconstruction driven by rollout step maps."""

from __future__ import annotations

import random


def add_stepmap_noise(
    input_ids: list[int],
    prompt_length: int,
    packed_step_maps: list[list[int]],
    mask_token_id: int,
    block_size: int,
    seed: int = 7,
) -> tuple[list[int], list[bool], list[int]]:
    """
    packed_step_maps covers the whole sequence as [original_id, step].
    Prompt entries use [-1, -1]. Blocks are aligned to the whole sequence.
    """
    if len(packed_step_maps) != len(input_ids):
        raise ValueError("packed_step_maps must align with the full sequence")
    if not 0 <= prompt_length <= len(input_ids):
        raise ValueError("prompt_length is outside the sequence")

    rng = random.Random(seed)
    # Prompt positions are stored as sentinels. For those positions, retain the
    # input token; generated positions carry the clean token returned by rollout.
    original_ids = [
        input_id if original_id <= 0 else original_id for input_id, (original_id, _) in zip(input_ids, packed_step_maps)
    ]
    steps = [entry[1] for entry in packed_step_maps]
    sequence_length = len(input_ids)
    response_length = sequence_length - prompt_length

    # Find the global block boundary immediately before the
    # response. It preserves the training model's absolute block alignment.
    first_response_block = prompt_length - prompt_length % block_size
    # An extra slot safely covers a final partial
    # block and is discarded later if it falls beyond the sequence boundary.
    block_count = response_length // block_size + 1
    thresholds: list[int] = []
    threshold_by_position = [1] * prompt_length

    for block_index in range(block_count):
        start = first_response_block + block_index * block_size
        end = min(start + block_size, sequence_length)
        while start < end - 1 and steps[start] <= 0:
            start += 1
        while end > start + 1 and steps[end - 1] <= 0:
            end -= 1
        if start >= sequence_length:
            break
        # Non-positive steps are prompt/padding sentinels, not diffusion steps.
        # Sampling between the observed minimum and maximum recreates one valid
        # intermediate denoising time for this absolute-position block.
        valid_steps = steps[start:end]
        threshold = rng.randint(min(valid_steps), max(valid_steps))
        thresholds.append(threshold)

        # Repeat each sampled value over a model block, prepend prompt
        # placeholders, then rolls by the prompt remainder.
        threshold_by_position.extend([threshold] * block_size)

    remainder = prompt_length % block_size
    if remainder:
        threshold_by_position = threshold_by_position[remainder:] + threshold_by_position[:remainder]
    threshold_by_position = threshold_by_position[:sequence_length]
    # A token revealed at this step or later must still be unknown at the sampled
    # state. This is the discrete forward process: select token positions first,
    # then replace their IDs with the mask token below.
    perturb_mask = [step >= max(threshold, 0) for step, threshold in zip(steps, threshold_by_position)]

    noisy_ids = [mask_token_id if masked else original_id for original_id, masked in zip(original_ids, perturb_mask)]
    return noisy_ids, perturb_mask, thresholds
