"""Pure selection and loss-weighting primitives for ESTF."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ESTFSelection:
    """Newly selected tokens, split by confidence and forced-fill origin."""

    selected: torch.Tensor
    confidence_selected: torch.Tensor
    forced_fill: torch.Tensor


def initialize_random_known_mask(
    eligible_mask: torch.Tensor,
    *,
    protected_mask: torch.Tensor,
    ratio_range: tuple[float, float],
    schedule_power: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the initially known subset while preserving protected positions."""
    if eligible_mask.ndim != 2 or protected_mask.shape != eligible_mask.shape:
        raise ValueError("eligible_mask and protected_mask must have shape [batch, sequence]")
    low, high = ratio_range
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("ratio_range must satisfy 0 <= low <= high <= 1")
    if schedule_power <= 0:
        raise ValueError("schedule_power must be positive")

    schedule = torch.rand(
        eligible_mask.shape,
        device=eligible_mask.device,
        generator=generator,
    ).pow(schedule_power)
    ratios = low + (high - low) * schedule
    selected = (
        torch.rand(
            eligible_mask.shape,
            device=eligible_mask.device,
            generator=generator,
        )
        < ratios
    )
    return selected & eligible_mask.bool() & ~protected_mask.bool()


def token_confidence(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    source: str,
) -> torch.Tensor:
    """Return either ground-truth probability or maximum model probability."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits and labels must have shapes [batch, sequence, vocabulary] and [batch, sequence]")
    probabilities = torch.softmax(logits.detach(), dim=-1, dtype=torch.float32)
    if source == "maximum":
        return probabilities.max(dim=-1).values
    if source != "ground_truth":
        raise ValueError("source must be 'ground_truth' or 'maximum'")
    safe_labels = labels.masked_fill(labels < 0, 0)
    return probabilities.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)


def select_estf_tokens(
    confidence: torch.Tensor,
    *,
    eligible_mask: torch.Tensor,
    known_mask: torch.Tensor,
    group_ids: torch.Tensor,
    stage: int,
    num_stages: int,
    confidence_threshold: float,
) -> ESTFSelection:
    """Select this stage's tokens and fill each block to its cumulative quota.

    Only valid, eligible tokens contribute to the number already known. This is
    intentional: padding or ignored labels must not make a partially valid
    block appear further through its curriculum than it really is.
    """
    expected_shape = confidence.shape
    if confidence.ndim != 2 or any(tensor.shape != expected_shape for tensor in (eligible_mask, known_mask, group_ids)):
        raise ValueError("confidence and curriculum tensors must have shape [batch, sequence]")
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    if not 0 <= stage < num_stages:
        raise ValueError("stage must satisfy 0 <= stage < num_stages")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")

    eligible = eligible_mask.bool()
    known = known_mask.bool()
    if torch.any(known & ~eligible):
        raise ValueError("known_mask must be a subset of eligible_mask")
    if torch.any(eligible & (group_ids < 0)):
        raise ValueError("every eligible token must have a non-negative group id")

    candidates = eligible & ~known
    confidence_selected = candidates & (confidence >= confidence_threshold)
    forced_fill = torch.zeros_like(candidates)

    for batch_index in range(confidence.shape[0]):
        batch_groups = torch.unique(group_ids[batch_index][eligible[batch_index]])
        for group_id in batch_groups:
            in_group = group_ids[batch_index] == group_id
            group_eligible = eligible[batch_index] & in_group
            group_known = known[batch_index] & group_eligible
            group_confident = confidence_selected[batch_index] & in_group

            total_count = int(group_eligible.sum().item())
            required_count = (total_count * (stage + 1) + num_stages - 1) // num_stages
            already_known_count = int(group_known.sum().item())
            confident_count = int(group_confident.sum().item())
            needed_count = max(required_count - already_known_count - confident_count, 0)
            if not needed_count:
                continue

            group_candidates = candidates[batch_index] & in_group & ~group_confident
            candidate_indices = torch.nonzero(group_candidates, as_tuple=False).flatten()
            selection_count = min(needed_count, candidate_indices.numel())
            if selection_count:
                scores = confidence[batch_index, candidate_indices]
                selected_indices = candidate_indices[torch.topk(scores, k=selection_count).indices]
                forced_fill[batch_index, selected_indices] = True

    selected = confidence_selected | forced_fill
    return ESTFSelection(
        selected=selected,
        confidence_selected=confidence_selected,
        forced_fill=forced_fill,
    )


def m2t_loss_weights(
    current_mask: torch.Tensor,
    *,
    eligible_mask: torch.Tensor,
    group_ids: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Scale remaining-token loss by the inverse masked ratio of each block."""
    expected_shape = current_mask.shape
    if current_mask.ndim != 2 or any(tensor.shape != expected_shape for tensor in (eligible_mask, group_ids)):
        raise ValueError("loss masks and group_ids must have shape [batch, sequence]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    current = current_mask.bool()
    eligible = eligible_mask.bool()
    if torch.any(current & ~eligible):
        raise ValueError("current_mask must be a subset of eligible_mask")
    if torch.any(eligible & (group_ids < 0)):
        raise ValueError("every eligible token must have a non-negative group id")

    weights = torch.zeros(expected_shape, dtype=torch.float32, device=current_mask.device)
    for batch_index in range(current.shape[0]):
        batch_groups = torch.unique(group_ids[batch_index][eligible[batch_index]])
        for group_id in batch_groups:
            group_current = current[batch_index] & (group_ids[batch_index] == group_id)
            current_count = int(group_current.sum().item())
            if current_count:
                weights[batch_index, group_current] = block_size / current_count
    return weights


__all__ = [
    "ESTFSelection",
    "initialize_random_known_mask",
    "m2t_loss_weights",
    "select_estf_tokens",
    "token_confidence",
]
