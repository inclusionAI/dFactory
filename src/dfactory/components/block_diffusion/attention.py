"""Dense Block Diffusion attention semantics used as an oracle and SDPA mask."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from dfactory.components.block_diffusion.layout import BlockLayout, build_paired_layout


@dataclass(frozen=True)
class SlidingWindowAttentionConfig:
    """Layer-aware sliding-window settings for Block Diffusion attention.

    ``window_blocks`` counts preceding blocks; the query's current block is
    visible in addition to that window. A scalar applies to every layer, while
    a tuple can use ``None`` to leave selected layers unrestricted.

    When ``activation_order`` is present, its first layer is windowed at step
    zero and one more layer is enabled after every ``activation_interval``.
    """

    window_blocks: int | tuple[int | None, ...]
    sink_tokens: int = 4
    activation_order: tuple[int, ...] = ()
    activation_interval: int = 500

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> SlidingWindowAttentionConfig:
        unknown = sorted(set(values) - {"window_blocks", "sink_tokens", "activation_order", "activation_interval"})
        if unknown:
            raise ValueError(f"unknown sliding-window attention parameter(s): {', '.join(unknown)}")
        normalized = dict(values)
        if isinstance(normalized.get("window_blocks"), list):
            normalized["window_blocks"] = tuple(normalized["window_blocks"])
        if "activation_order" in normalized:
            normalized["activation_order"] = tuple(normalized["activation_order"])
        try:
            config = cls(**normalized)
        except TypeError as exc:
            raise ValueError(f"invalid sliding-window attention config: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if type(self.window_blocks) is int:
            windows = (self.window_blocks,)
        elif isinstance(self.window_blocks, tuple):
            windows = self.window_blocks
        else:
            raise ValueError("window_blocks must be a positive integer or a list of layer settings")
        if not windows:
            raise ValueError("window_blocks must contain at least one layer setting")
        if any(window is not None and (type(window) is not int or window <= 0) for window in windows):
            raise ValueError("window_blocks values must be positive integers or null")
        if type(self.sink_tokens) is not int or self.sink_tokens < 0:
            raise ValueError("sink_tokens must be a non-negative integer")
        if type(self.activation_interval) is not int or self.activation_interval <= 0:
            raise ValueError("activation_interval must be a positive integer")
        if not isinstance(self.activation_order, tuple):
            raise ValueError("activation_order must be a list of layer indices")
        if any(type(layer) is not int or layer < 0 for layer in self.activation_order):
            raise ValueError("activation_order must contain non-negative layer indices")
        if len(set(self.activation_order)) != len(self.activation_order):
            raise ValueError("activation_order must not contain duplicate layers")
        if type(self.window_blocks) is not int and any(
            layer >= len(self.window_blocks) for layer in self.activation_order
        ):
            raise ValueError("activation_order contains a layer not defined by window_blocks")

    @property
    def unique_windows(self) -> tuple[int, ...]:
        windows = (self.window_blocks,) if type(self.window_blocks) is int else self.window_blocks
        return tuple(sorted({window for window in windows if window is not None}))

    def window_for_layer(self, layer_index: int, *, training_step: int) -> int | None:
        if layer_index < 0:
            raise ValueError("layer_index must not be negative")
        if training_step < 0:
            raise ValueError("training_step must not be negative")
        if self.activation_order:
            active_count = min(training_step // self.activation_interval + 1, len(self.activation_order))
            if layer_index not in self.activation_order[:active_count]:
                return None
        if type(self.window_blocks) is int:
            return self.window_blocks
        if layer_index >= len(self.window_blocks):
            raise ValueError(
                f"window_blocks defines {len(self.window_blocks)} layers but layer {layer_index} was requested"
            )
        return self.window_blocks[layer_index]


@dataclass(frozen=True)
class BlockDiffusionAttentionMasks:
    """Base attention plus reusable window variants selected by a model layer."""

    base: Any
    sliding_windows: dict[int, Any]
    sliding_window_config: SlidingWindowAttentionConfig | None = None

    def for_layer(self, layer_index: int, *, training_step: int = 0) -> Any:
        if self.sliding_window_config is None:
            return self.base
        window = self.sliding_window_config.window_for_layer(layer_index, training_step=training_step)
        return self.base if window is None else self.sliding_windows[window]


def _attention_relations(
    layout: BlockLayout,
    *,
    window_blocks: int | None = None,
    sink_tokens: int = 0,
) -> torch.Tensor:
    if window_blocks is not None and window_blocks <= 0:
        raise ValueError("window_blocks must be positive")
    if sink_tokens < 0:
        raise ValueError("sink_tokens must not be negative")

    q_pair = layout.pair_ids.unsqueeze(-1)
    kv_pair = layout.pair_ids.unsqueeze(-2)
    same_pair = q_pair == kv_pair

    q_type = layout.slot_types.unsqueeze(-1)
    kv_type = layout.slot_types.unsqueeze(-2)
    q_block = layout.local_block_ids.unsqueeze(-1)
    kv_block = layout.local_block_ids.unsqueeze(-2)

    noisy_to_noisy = (q_type == 0) & (kv_type == 0) & (q_block == kv_block)
    noisy_to_clean = (q_type == 0) & (kv_type == 1) & (q_block > kv_block)
    clean_to_clean = (q_type == 1) & (kv_type == 1) & (q_block >= kv_block)
    if window_blocks is not None:
        in_window = kv_block > q_block - window_blocks - 1
        sink_key = (kv_type == 1) & (layout.position_ids.unsqueeze(-2) < sink_tokens)
        noisy_to_clean &= in_window | sink_key
        clean_to_clean &= in_window | sink_key
    valid = layout.valid_mask.unsqueeze(-1) & layout.valid_mask.unsqueeze(-2)
    return same_pair & (noisy_to_noisy | noisy_to_clean | clean_to_clean) & valid


def build_attention_mask(layout: BlockLayout) -> torch.Tensor:
    """Return a boolean mask where ``True`` means a query may attend to a key."""
    return _attention_relations(layout)


def build_sliding_window_attention_mask(
    layout: BlockLayout,
    *,
    window_blocks: int,
    sink_tokens: int = 4,
) -> torch.Tensor:
    """Return Block Diffusion attention constrained by a local clean-context window."""
    return _attention_relations(layout, window_blocks=window_blocks, sink_tokens=sink_tokens)


def build_flex_attention_mask(
    layout: BlockLayout,
    *,
    window_blocks: int | None = None,
    sink_tokens: int = 0,
):
    """Build the sparse mask consumed by PyTorch FlexAttention."""
    from torch.nn.attention.flex_attention import create_block_mask

    if window_blocks is not None and window_blocks <= 0:
        raise ValueError("window_blocks must be positive")
    if sink_tokens < 0:
        raise ValueError("sink_tokens must not be negative")

    batch_size, sequence_length = layout.pair_ids.shape

    def mask_mod(batch_index, _head_index, query_index, key_index):
        same_pair = layout.pair_ids[batch_index, query_index] == layout.pair_ids[batch_index, key_index]
        query_type = layout.slot_types[batch_index, query_index]
        key_type = layout.slot_types[batch_index, key_index]
        query_block = layout.local_block_ids[batch_index, query_index]
        key_block = layout.local_block_ids[batch_index, key_index]
        noisy_to_noisy = (query_type == 0) & (key_type == 0) & (query_block == key_block)
        noisy_to_clean = (query_type == 0) & (key_type == 1) & (query_block > key_block)
        clean_to_clean = (query_type == 1) & (key_type == 1) & (query_block >= key_block)
        if window_blocks is not None:
            in_window = key_block > query_block - window_blocks - 1
            sink_key = (key_type == 1) & (layout.position_ids[batch_index, key_index] < sink_tokens)
            noisy_to_clean = noisy_to_clean & (in_window | sink_key)
            clean_to_clean = clean_to_clean & (in_window | sink_key)
        valid = layout.valid_mask[batch_index, query_index] & layout.valid_mask[batch_index, key_index]
        return same_pair & (noisy_to_noisy | noisy_to_clean | clean_to_clean) & valid

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=str(layout.pair_ids.device),
        _compile=False,
    )


def build_block_diffusion_attention_masks(
    layout: BlockLayout,
    *,
    backend: str,
    sliding_window: SlidingWindowAttentionConfig | None = None,
) -> BlockDiffusionAttentionMasks:
    """Build a base mask and each distinct layer window exactly once."""
    if backend == "flex_attention":
        base = build_flex_attention_mask(layout)
        windows = {
            window: build_flex_attention_mask(
                layout,
                window_blocks=window,
                sink_tokens=sliding_window.sink_tokens,
            )
            for window in (() if sliding_window is None else sliding_window.unique_windows)
        }
    elif backend in {"eager", "sdpa"}:
        allowed = build_attention_mask(layout)
        base = torch.zeros(
            (*allowed.shape[:1], 1, *allowed.shape[1:]),
            dtype=torch.float32,
            device=allowed.device,
        ).masked_fill_(~allowed.unsqueeze(1), float("-inf"))
        windows = {}
        if sliding_window is not None:
            for window in sliding_window.unique_windows:
                allowed_window = build_sliding_window_attention_mask(
                    layout,
                    window_blocks=window,
                    sink_tokens=sliding_window.sink_tokens,
                )
                windows[window] = torch.zeros_like(base).masked_fill_(
                    ~allowed_window.unsqueeze(1),
                    float("-inf"),
                )
    else:
        raise ValueError(f"unsupported attention backend: {backend}")
    return BlockDiffusionAttentionMasks(
        base=base,
        sliding_windows=windows,
        sliding_window_config=sliding_window,
    )


def build_single_pair_attention_mask(sequence_length: int, block_size: int) -> torch.Tensor:
    layout = build_paired_layout([[sequence_length]], block_size=block_size)
    return build_attention_mask(layout)[0]


__all__ = [
    "BlockDiffusionAttentionMasks",
    "SlidingWindowAttentionConfig",
    "build_attention_mask",
    "build_block_diffusion_attention_masks",
    "build_flex_attention_mask",
    "build_single_pair_attention_mask",
    "build_sliding_window_attention_mask",
]
