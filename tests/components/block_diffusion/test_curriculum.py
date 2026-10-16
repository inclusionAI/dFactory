import torch

from dfactory.components.block_diffusion import select_curriculum_tokens


def test_curriculum_selects_a_quota_per_block_and_finishes_coverage() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 0.0],
                [6.0, 0.0],
                [0.0, 6.0],
                [0.0, 0.0],
            ]
        ]
    )
    eligible = torch.ones((1, 4), dtype=torch.bool)
    history = torch.zeros_like(eligible)
    group_ids = torch.tensor([[0, 0, 1, 1]])

    first = select_curriculum_tokens(
        logits,
        eligible_mask=eligible,
        selected_history=history,
        group_ids=group_ids,
        stage=0,
        num_stages=2,
        confidence_threshold=0.9,
    )

    assert first.tolist() == [[False, True, True, False]]
    assert not history.any()

    second = select_curriculum_tokens(
        logits,
        eligible_mask=eligible,
        selected_history=first,
        group_ids=group_ids,
        stage=1,
        num_stages=2,
        confidence_threshold=0.9,
    )

    assert second.tolist() == [[True, False, False, True]]
    assert torch.all(first | second)


def test_high_confidence_tokens_expand_the_stage_quota() -> None:
    logits = torch.tensor([[[8.0, 0.0], [0.0, 8.0], [0.0, 0.0], [0.0, 0.0]]])
    eligible = torch.ones((1, 4), dtype=torch.bool)

    selected = select_curriculum_tokens(
        logits,
        eligible_mask=eligible,
        selected_history=torch.zeros_like(eligible),
        group_ids=torch.zeros((1, 4), dtype=torch.long),
        stage=0,
        num_stages=4,
        confidence_threshold=0.9,
    )

    assert selected.tolist() == [[True, True, False, False]]
