"""Composable Block Diffusion primitives; complete algorithms live in recipes."""

from dfactory.components.block_diffusion.attention import (
    BlockDiffusionAttentionMasks,
    SlidingWindowAttentionConfig,
    build_attention_mask,
    build_block_diffusion_attention_masks,
    build_flex_attention_mask,
    build_sliding_window_attention_mask,
)
from dfactory.components.block_diffusion.curriculum import select_curriculum_tokens
from dfactory.components.block_diffusion.layout import BlockLayout, build_paired_layout, interleave_paired_slots
from dfactory.components.block_diffusion.noise import NoiseResult, apply_block_noise
from dfactory.components.block_diffusion.objectives import (
    ObjectiveResult,
    row_mean_same_token_cross_entropy,
    same_token_cross_entropy,
    shifted_cross_entropy,
)

__all__ = [
    "BlockLayout",
    "BlockDiffusionAttentionMasks",
    "NoiseResult",
    "ObjectiveResult",
    "SlidingWindowAttentionConfig",
    "apply_block_noise",
    "build_attention_mask",
    "build_block_diffusion_attention_masks",
    "build_flex_attention_mask",
    "build_paired_layout",
    "build_sliding_window_attention_mask",
    "interleave_paired_slots",
    "row_mean_same_token_cross_entropy",
    "same_token_cross_entropy",
    "select_curriculum_tokens",
    "shifted_cross_entropy",
]
