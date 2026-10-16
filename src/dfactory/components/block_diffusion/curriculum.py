"""Pure curriculum-selection primitives for multi-stage Block Diffusion recipes."""

from __future__ import annotations

import torch


def select_curriculum_tokens(
    logits: torch.Tensor,
    *,
    eligible_mask: torch.Tensor,
    selected_history: torch.Tensor,
    group_ids: torch.Tensor,
    stage: int,
    num_stages: int,
    confidence_threshold: float,
) -> torch.Tensor:
    """Select the next per-block token quota without mutating recipe state.

    The cumulative quota grows linearly with the stage. Tokens above the
    confidence threshold expand that quota, and the final stage always covers
    every remaining eligible token.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    expected_shape = logits.shape[:2]
    if any(tensor.shape != expected_shape for tensor in (eligible_mask, selected_history, group_ids)):
        raise ValueError("curriculum masks and group_ids must match the first two logits dimensions")
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    if not 0 <= stage < num_stages:
        raise ValueError("stage must satisfy 0 <= stage < num_stages")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")

    eligible = eligible_mask.bool()
    history = selected_history.bool()
    if torch.any(history & ~eligible):
        raise ValueError("selected_history must be a subset of eligible_mask")
    if torch.any(eligible & (group_ids < 0)):
        raise ValueError("every eligible token must have a non-negative group id")

    candidates = eligible & ~history
    selected = torch.zeros_like(candidates)
    if not torch.any(candidates):
        return selected

    confidence = torch.full(
        expected_shape,
        -1.0,
        dtype=logits.dtype,
        device=logits.device,
    )
    candidate_logits = logits.detach()[candidates]
    max_logits = candidate_logits.max(dim=-1).values
    confidence[candidates] = (max_logits - torch.logsumexp(candidate_logits, dim=-1)).exp()

    for batch_index in range(logits.shape[0]):
        batch_groups = torch.unique(group_ids[batch_index][eligible[batch_index]])
        for group_id in batch_groups:
            in_group = group_ids[batch_index] == group_id
            group_eligible = eligible[batch_index] & in_group
            group_history = history[batch_index] & in_group
            group_candidates = candidates[batch_index] & in_group
            candidate_indices = torch.nonzero(group_candidates, as_tuple=False).flatten()
            if not candidate_indices.numel():
                continue

            total_count = int(group_eligible.sum().item())
            history_count = int(group_history.sum().item())
            target_count = total_count * (stage + 1) // num_stages
            base_count = max(target_count - history_count, 0)
            candidate_confidence = confidence[batch_index, candidate_indices]
            high_confidence_count = int((candidate_confidence > confidence_threshold).sum().item())
            selection_count = min(
                max(base_count, high_confidence_count),
                candidate_indices.numel(),
            )
            if selection_count:
                selected_indices = candidate_indices[torch.topk(candidate_confidence, k=selection_count).indices]
                selected[batch_index, selected_indices] = True

    return selected


__all__ = ["select_curriculum_tokens"]
