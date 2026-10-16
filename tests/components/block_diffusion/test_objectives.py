import torch
import torch.nn.functional as functional

from dfactory.components.block_diffusion.objectives import (
    row_mean_same_token_cross_entropy,
    same_token_cross_entropy,
    shifted_cross_entropy,
)


def test_empty_same_token_mask_returns_zero_with_gradient() -> None:
    logits = torch.randn(1, 3, 5, requires_grad=True)
    result = same_token_cross_entropy(logits, torch.full((1, 3), -100))
    result.loss.backward()
    assert result.loss.item() == 0.0
    assert logits.grad is not None


def test_same_token_loss_normalizes_each_batch_row_before_averaging() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]],
        ]
    )
    labels = torch.tensor([[1, -100, -100], [1, 1, 1]])

    result = row_mean_same_token_cross_entropy(logits, labels)

    first_row = functional.cross_entropy(logits[0, :1], labels[0, :1])
    second_row = functional.cross_entropy(logits[1], labels[1])
    assert torch.allclose(result.loss, (first_row + second_row) / 2)
    assert result.token_count.item() == 4


def test_shared_same_token_loss_averages_all_selected_tokens() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]],
        ]
    )
    labels = torch.tensor([[1, -100, -100], [1, 1, 1]])

    result = same_token_cross_entropy(logits, labels)

    selected = labels != -100
    expected = functional.cross_entropy(logits[selected], labels[selected])
    assert torch.allclose(result.loss, expected)


def test_shifted_loss_uses_next_position_labels() -> None:
    logits = torch.full((1, 3, 4), -5.0)
    logits[0, 0, 2] = 5.0
    labels = torch.tensor([[-100, 2, -100]])
    result = shifted_cross_entropy(logits, labels)
    assert result.token_count.item() == 1
    assert result.loss.item() < 0.001
