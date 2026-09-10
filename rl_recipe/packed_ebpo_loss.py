"""Single-sequence, dependency-free rendering of PackedEBPOLoss's core math."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LossResult:
    current_elbo: float
    old_elbo: float
    block_ratios: tuple[float, ...]
    clipped_block_ratios: tuple[float, ...]
    ratio: float
    clipped_ratio: float
    policy_loss: float
    kl_loss: float
    total_loss: float


def _block_means(values: list[float], active: list[bool], block_size: int) -> list[float]:
    means = []
    for start in range(0, len(values), block_size):
        # effective_mask = completion_mask * perturb_mask: 
        # only generated tokens that are masked in x_t contribute to the ELBO estimate.
        selected = [
            value
            for value, enabled in zip(
                values[start : start + block_size], active[start : start + block_size]
            )
            if enabled
        ]
        if selected:
            means.append(sum(selected) / len(selected))
    if not means:
        raise ValueError("the sequence must contain at least one active response token")
    return means


def packed_ebpo_loss(
    current_logprobs: list[float],
    old_logprobs: list[float],
    ref_logprobs: list[float],
    completion_mask: list[bool],
    perturb_mask: list[bool],
    advantage: float,
    block_size: int = 4,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    kl_coefficient: float = 1e-4,
) -> LossResult:
    """Demonstrate the configured block-wise branch of PackedEBPOLoss.

    A complete training stack first gathers packed model logits. This demo begins at
    the resulting token log-probs and preserves the essential loss aggregation.
    """
    size = len(current_logprobs)
    if not all(len(values) == size for values in (old_logprobs, ref_logprobs, completion_mask, perturb_mask)):
        raise ValueError("all per-token inputs must have equal length")

    active = [completion and perturbed for completion, perturbed in zip(completion_mask, perturb_mask)]
    # First average target-token log-probabilities inside each valid diffusion
    # block. Empty blocks are excluded through the `valid_blk` mask.
    current_blocks = _block_means(current_logprobs, active, block_size)
    old_blocks = _block_means(old_logprobs, active, block_size)
    ref_blocks = _block_means(ref_logprobs, active, block_size)

    # The sequence ELBO is the mean of valid block ELBOs. It is retained for
    # reporting; the configured optimization branch computes ratios per block.
    current_elbo = sum(current_blocks) / len(current_blocks)
    old_elbo = sum(old_blocks) / len(old_blocks)

    # Stabilized importance sampling ratio (block-wise branch):
    # rho_b = exp(ELBO_theta,b - ELBO_old,b), with log-ratio safety clamping.
    block_ratios = tuple(
        math.exp(max(-10.0, min(10.0, current_value - old_value)))
        for current_value, old_value in zip(current_blocks, old_blocks)
    )
    clipped_block_ratios = tuple(
        max(1.0 - clip_low, min(1.0 + clip_high, ratio))
        for ratio in block_ratios
    )
    # PPO uses the smaller of the unclipped and clipped surrogate. This single
    # expression also preserves the correct clipping direction for negative A.
    block_surrogates = [
        min(ratio * advantage, clipped * advantage)
        for ratio, clipped in zip(block_ratios, clipped_block_ratios)
    ]
    policy_loss = -sum(block_surrogates) / len(block_surrogates)

    # Compute block-level KL regularization between the current and reference
    # policies. Scale block means back to block-level values before aggregation.
    kl_terms = [0.5 * ((cur - ref) * block_size) ** 2 for cur, ref in zip(current_blocks, ref_blocks)]
    kl_loss = sum(kl_terms) / len(kl_terms)
    total_loss = policy_loss + kl_coefficient * kl_loss
    ratio = sum(block_ratios) / len(block_ratios)
    clipped_ratio = sum(clipped_block_ratios) / len(clipped_block_ratios)
    return LossResult(
        current_elbo,
        old_elbo,
        block_ratios,
        clipped_block_ratios,
        ratio,
        clipped_ratio,
        policy_loss,
        kl_loss,
        total_loss,
    )
