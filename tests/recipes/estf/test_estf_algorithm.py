import pytest
import torch

from dfactory.recipes.estf.algorithm import (
    initialize_random_known_mask,
    m2t_loss_weights,
    select_estf_tokens,
    token_confidence,
)


def test_random_known_initialization_keeps_block_ends_masked() -> None:
    eligible = torch.tensor([[True, True, True, True, True, False]])
    block_ends = torch.tensor([[False, True, False, True, False, True]])

    known = initialize_random_known_mask(
        eligible,
        protected_mask=block_ends,
        ratio_range=(1.0, 1.0),
        schedule_power=2.0,
    )

    assert known.tolist() == [[True, False, True, False, True, False]]


def test_forced_fill_counts_only_valid_known_tokens_in_each_block() -> None:
    eligible = torch.tensor([[True, True, True, True, True, True, False, False]])
    group_ids = torch.zeros((1, 8), dtype=torch.long)
    confidence = torch.arange(8, dtype=torch.float32).unsqueeze(0) / 100

    selection = select_estf_tokens(
        confidence,
        eligible_mask=eligible,
        known_mask=torch.zeros_like(eligible),
        group_ids=group_ids,
        stage=0,
        num_stages=4,
        confidence_threshold=0.85,
    )

    assert selection.confidence_selected.sum().item() == 0
    assert selection.forced_fill.sum().item() == 2
    assert selection.selected[0, :6].sum().item() == 2
    assert not torch.any(selection.selected[0, 6:])


def test_forced_fill_advances_every_block_to_its_iteration_quota() -> None:
    eligible = torch.tensor([[True, True, True, True, True, True, False, False]])
    group_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    confidence = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.0]])
    empty = torch.zeros_like(eligible)

    first = select_estf_tokens(
        confidence,
        eligible_mask=eligible,
        known_mask=empty,
        group_ids=group_ids,
        stage=0,
        num_stages=4,
        confidence_threshold=0.85,
    )
    second = select_estf_tokens(
        confidence,
        eligible_mask=eligible,
        known_mask=first.selected,
        group_ids=group_ids,
        stage=1,
        num_stages=4,
        confidence_threshold=0.85,
    )

    assert first.forced_fill.sum().item() == 2
    assert first.selected[0, :4].sum().item() == 1
    assert first.selected[0, 4:].sum().item() == 1
    assert second.forced_fill.sum().item() == 1
    assert (first.selected | second.selected)[0, :4].sum().item() == 2
    assert (first.selected | second.selected)[0, 4:].sum().item() == 1


def test_m2t_loss_weights_use_each_blocks_remaining_ratio() -> None:
    eligible = torch.tensor([[True, True, True, True, True, True, False, False]])
    current = torch.tensor([[True, True, True, True, True, False, False, False]])
    group_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])

    weights = m2t_loss_weights(
        current,
        eligible_mask=eligible,
        group_ids=group_ids,
        block_size=4,
    )

    assert weights.tolist() == [[1.0, 1.0, 1.0, 1.0, 4.0, 0.0, 0.0, 0.0]]


def test_token_confidence_supports_ground_truth_and_maximum_probability() -> None:
    logits = torch.tensor([[[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]]])
    labels = torch.tensor([[2, -100]])

    ground_truth = token_confidence(logits, labels, source="ground_truth")
    maximum = token_confidence(logits, labels, source="maximum")

    assert ground_truth[0, 0] == pytest.approx(torch.softmax(logits[0, 0], dim=-1)[2].item())
    assert maximum[0, 0] == pytest.approx(torch.softmax(logits[0, 0], dim=-1)[1].item())
    with pytest.raises(ValueError, match="source must be"):
        token_confidence(logits, labels, source="entropy")
