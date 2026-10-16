import torch

from dfactory.components.block_diffusion.attention import (
    SlidingWindowAttentionConfig,
    build_attention_mask,
    build_block_diffusion_attention_masks,
    build_flex_attention_mask,
    build_sliding_window_attention_mask,
)
from dfactory.components.block_diffusion.layout import build_paired_layout, interleave_paired_slots


def test_single_pair_attention_relations() -> None:
    layout = build_paired_layout([[4]], block_size=2)
    mask = build_attention_mask(layout)[0]

    assert mask[0, 1]
    assert not mask[0, 2]
    assert not mask[0, 4]
    assert mask[2, 4]
    assert not mask[4, 0]
    assert mask[6, 4]


def test_packed_pairs_are_isolated_and_padding_is_hidden() -> None:
    layout = build_paired_layout([[2, 2]], block_size=2, total_length=10, valid_lengths=[[2, 1]])
    mask = build_attention_mask(layout)[0]

    assert not mask[:4, 4:8].any()
    assert not mask[4:8, :4].any()
    assert not mask[5].any()
    assert not mask[:, 5].any()
    assert not mask[8:].any()
    assert not mask[:, 8:].any()


def test_layout_rejects_non_aligned_lengths() -> None:
    try:
        build_paired_layout(torch.tensor([[3]]), block_size=2)
    except ValueError as exc:
        assert "block aligned" in str(exc)
    else:
        raise AssertionError("expected non-aligned sequence length to fail")


def test_flex_attention_mask_preserves_token_level_semantics() -> None:
    layout = build_paired_layout([[128]], block_size=32)
    block_mask = build_flex_attention_mask(layout)

    assert block_mask.shape == (1, 1, 256, 256)
    assert block_mask.mask_mod(0, 0, 0, 1)
    assert not block_mask.mask_mod(0, 0, 0, 128)
    assert block_mask.mask_mod(0, 0, 32, 128)


def test_sliding_window_limits_clean_history_and_preserves_sink_tokens() -> None:
    layout = build_paired_layout([[8]], block_size=2)
    mask = build_sliding_window_attention_mask(layout, window_blocks=1, sink_tokens=1)[0]

    # The last clean block sees its current and preceding block, plus the first
    # clean-slot token as an attention sink.
    assert mask[14, 12]
    assert mask[14, 8]
    assert not mask[14, 9]
    # Noisy-to-noisy attention remains block local; noisy-to-clean uses the
    # same window and clean-slot sink semantics.
    assert mask[6, 7]
    assert mask[6, 8]
    assert not mask[6, 9]


def test_sliding_window_is_local_to_each_packed_pair() -> None:
    layout = build_paired_layout([[4, 4]], block_size=2, valid_lengths=[[4, 3]])
    mask = build_sliding_window_attention_mask(layout, window_blocks=1, sink_tokens=1)[0]

    assert not mask[:8, 8:].any()
    assert not mask[8:, :8].any()
    assert not mask[11].any()
    assert not mask[:, 11].any()


def test_flex_sliding_window_matches_dense_relations() -> None:
    layout = build_paired_layout([[128]], block_size=32)
    dense = build_sliding_window_attention_mask(layout, window_blocks=1, sink_tokens=1)
    flex = build_flex_attention_mask(layout, window_blocks=1, sink_tokens=1)

    for query, key in ((224, 192), (224, 128), (224, 129), (96, 97), (96, 128), (96, 129)):
        assert bool(flex.mask_mod(0, 0, query, key)) is bool(dense[0, query, key])


def test_layer_attention_masks_support_fixed_and_progressive_activation() -> None:
    layout = build_paired_layout([[4]], block_size=2)
    fixed = SlidingWindowAttentionConfig(window_blocks=(1, None), sink_tokens=1)
    masks = build_block_diffusion_attention_masks(layout, backend="sdpa", sliding_window=fixed)

    assert masks.for_layer(0, training_step=0) is masks.sliding_windows[1]
    assert masks.for_layer(1, training_step=0) is masks.base

    progressive = SlidingWindowAttentionConfig(
        window_blocks=1,
        activation_order=(1, 0),
        activation_interval=10,
    )
    masks = build_block_diffusion_attention_masks(layout, backend="sdpa", sliding_window=progressive)

    assert masks.for_layer(1, training_step=0) is masks.sliding_windows[1]
    assert masks.for_layer(0, training_step=0) is masks.base
    assert masks.for_layer(0, training_step=10) is masks.sliding_windows[1]


def test_sliding_window_config_rejects_invalid_values() -> None:
    for values, message in (
        ({"window_blocks": 0}, "positive integer"),
        ({"window_blocks": True}, "positive integer"),
        ({"window_blocks": 1, "sink_tokens": -1}, "non-negative integer"),
        ({"window_blocks": 1, "activation_order": [0, 0]}, "duplicate"),
    ):
        try:
            SlidingWindowAttentionConfig.from_mapping(values)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("expected invalid sliding-window config to fail")


def test_packed_values_are_interleaved_by_pair() -> None:
    values = torch.tensor([[1, 2, 3, 4]])
    paired = interleave_paired_slots(values, [[2, 2]], fill_value=0)
    assert paired.tolist() == [[1, 2, 1, 2, 3, 4, 3, 4]]
