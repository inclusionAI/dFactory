"""Document-isolated causal attention used by autoregressive SFT."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DocumentLayout:
    document_ids: torch.Tensor
    position_ids: torch.Tensor
    valid_mask: torch.Tensor


def build_document_layout(
    sequence_lengths: torch.Tensor | Sequence[Sequence[int]],
    *,
    total_length: int | None = None,
    valid_lengths: torch.Tensor | Sequence[Sequence[int]] | None = None,
) -> DocumentLayout:
    """Describe concatenated documents without allowing padding to become a document."""
    lengths = torch.as_tensor(sequence_lengths, dtype=torch.long)
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(0)
    if lengths.ndim != 2:
        raise ValueError("sequence_lengths must have shape [batch, documents]")
    if torch.any(lengths < 0):
        raise ValueError("sequence lengths must not be negative")

    valid = (
        lengths if valid_lengths is None else torch.as_tensor(valid_lengths, dtype=torch.long, device=lengths.device)
    )
    if valid.shape != lengths.shape:
        raise ValueError("valid_lengths must have the same shape as sequence_lengths")
    if torch.any(valid < 0) or torch.any(valid > lengths):
        raise ValueError("valid_lengths must satisfy 0 <= valid length <= sequence length")

    required_lengths = lengths.sum(dim=1)
    inferred_length = int(required_lengths.max().item()) if required_lengths.numel() else 0
    total_length = inferred_length if total_length is None else total_length
    if total_length < inferred_length:
        raise ValueError(f"total_length={total_length} is smaller than required length {inferred_length}")

    batch_size = lengths.shape[0]
    device = lengths.device
    document_ids = torch.zeros((batch_size, total_length), dtype=torch.long, device=device)
    position_ids = torch.zeros_like(document_ids)
    valid_mask = torch.zeros((batch_size, total_length), dtype=torch.bool, device=device)

    for batch_index, sample_lengths in enumerate(lengths.tolist()):
        offset = 0
        for document_index, length in enumerate(sample_lengths):
            if length == 0:
                continue
            end = offset + length
            valid_length = int(valid[batch_index, document_index].item())
            position_ids[batch_index, offset:end] = torch.arange(length, device=device)
            document_ids[batch_index, offset : offset + valid_length] = document_index + 1
            valid_mask[batch_index, offset : offset + valid_length] = True
            offset = end

    return DocumentLayout(document_ids=document_ids, position_ids=position_ids, valid_mask=valid_mask)


def build_attention_mask(layout: DocumentLayout) -> torch.Tensor:
    """Return a boolean causal mask with complete isolation between documents."""
    query_documents = layout.document_ids.unsqueeze(-1)
    key_documents = layout.document_ids.unsqueeze(-2)
    same_document = (query_documents == key_documents) & (query_documents > 0)
    query_positions = layout.position_ids.unsqueeze(-1)
    key_positions = layout.position_ids.unsqueeze(-2)
    causal = query_positions >= key_positions
    valid = layout.valid_mask.unsqueeze(-1) & layout.valid_mask.unsqueeze(-2)
    return same_document & causal & valid


def build_flex_attention_mask(layout: DocumentLayout):
    """Build the sparse document-causal mask consumed by PyTorch FlexAttention."""
    from torch.nn.attention.flex_attention import create_block_mask

    batch_size, sequence_length = layout.document_ids.shape

    def mask_mod(batch_index, _head_index, query_index, key_index):
        query_document = layout.document_ids[batch_index, query_index]
        key_document = layout.document_ids[batch_index, key_index]
        same_document = (query_document == key_document) & (query_document > 0)
        causal = layout.position_ids[batch_index, query_index] >= layout.position_ids[batch_index, key_index]
        valid = layout.valid_mask[batch_index, query_index] & layout.valid_mask[batch_index, key_index]
        return same_document & causal & valid

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=str(layout.document_ids.device),
        _compile=False,
    )


__all__ = ["DocumentLayout", "build_attention_mask", "build_document_layout", "build_flex_attention_mask"]
