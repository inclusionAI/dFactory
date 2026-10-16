import torch

from dfactory.components.block_diffusion.layout import build_paired_layout
from dfactory.components.block_diffusion.noise import apply_block_noise


def test_noise_only_changes_trainable_noisy_slots() -> None:
    layout = build_paired_layout([[4]], block_size=2)
    input_ids = torch.arange(8).unsqueeze(0)
    labels = input_ids.clone()
    trainable = torch.ones_like(input_ids, dtype=torch.bool)

    result = apply_block_noise(
        input_ids,
        labels,
        trainable_mask=trainable,
        noisy_slot_mask=layout.noisy_slot_mask,
        noise_group_ids=layout.noise_group_ids,
        mask_token_id=99,
        noise_range=(1.0, 1.0),
    )
    assert result.input_ids.tolist() == [[99, 99, 99, 99, 4, 5, 6, 7]]
    assert result.mask.tolist() == [[True, True, True, True, False, False, False, False]]


def test_zero_noise_produces_differentiable_empty_objective_inputs() -> None:
    layout = build_paired_layout([[2]], block_size=2)
    input_ids = torch.tensor([[1, 2, 1, 2]])
    labels = input_ids.clone()
    result = apply_block_noise(
        input_ids,
        labels,
        trainable_mask=torch.ones_like(input_ids, dtype=torch.bool),
        noisy_slot_mask=layout.noisy_slot_mask,
        noise_group_ids=layout.noise_group_ids,
        mask_token_id=99,
        noise_range=(0.0, 0.0),
    )
    assert not result.mask.any()
    assert result.labels[0, :2].tolist() == [-100, -100]


def test_multiturn_transition_hides_later_non_trainable_tokens_in_block() -> None:
    layout = build_paired_layout([[4]], block_size=4)
    input_ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]])
    trainable = torch.tensor([[False, True, False, False, False, False, False, False]])
    labels = input_ids.masked_fill(~trainable, -100)

    result = apply_block_noise(
        input_ids,
        labels,
        trainable_mask=trainable,
        noisy_slot_mask=layout.noisy_slot_mask,
        noise_group_ids=layout.noise_group_ids,
        mask_token_id=15,
        eos_token_id=14,
        noise_range=(0.0, 0.0),
    )

    assert result.input_ids.tolist() == [[1, 2, 14, 14, 1, 2, 3, 4]]
    assert not result.mask.any()
