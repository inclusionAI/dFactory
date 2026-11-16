"""EOS-isolated block diffusion attention for SolDiff pretraining."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask


def block_diffusion_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """

    # Indicate whether token belongs to xt or x0
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    # Compute block indices
    block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal


def packed_block_mask_mod(b, h, q_idx, kv_idx, boolean_mask):
    return boolean_mask[b, q_idx, kv_idx]


def build_doc_ids_from_eos(clean_input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    eos_before_current_token = F.pad(
        (clean_input_ids[:, :-1] == eos_token_id).to(torch.long),
        (1, 0),
        value=0,
    )
    document_ids = torch.cumsum(eos_before_current_token, dim=1)
    return document_ids


def build_clean_blockwise_doc_attn_mask(
    clean_input_ids: torch.Tensor,
    block_size: int,
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length = clean_input_ids.shape
    document_ids = build_doc_ids_from_eos(clean_input_ids, eos_token_id)

    token_positions = torch.arange(sequence_length, device=clean_input_ids.device)
    physical_block_ids = torch.div(token_positions, block_size, rounding_mode="floor")

    query_block_ids = physical_block_ids.view(1, sequence_length, 1).expand(batch_size, -1, -1)
    key_block_ids = physical_block_ids.view(1, 1, sequence_length).expand(batch_size, -1, -1)

    same_document_flag = document_ids.unsqueeze(2) == document_ids.unsqueeze(1)
    blockwise_causal_flag = query_block_ids >= key_block_ids
    allow_attention_flag = same_document_flag & blockwise_causal_flag
    disallow_attention_flag = ~allow_attention_flag
    return disallow_attention_flag.unsqueeze(1), document_ids


def build_attention_mask_by_impl(
    disallow_attention_flag: torch.Tensor,
    attn_implementation: str,
    enable_mixed_precision: bool,
) -> torch.Tensor:
    batch_size = disallow_attention_flag.shape[0]
    query_length = disallow_attention_flag.shape[-2]
    key_length = disallow_attention_flag.shape[-1]

    if attn_implementation == "flex_attention":
        boolean_allow_attention = disallow_attention_flag.bool().logical_not().squeeze(1).contiguous()
        block_mask_function = partial(
            packed_block_mask_mod,
            boolean_mask=boolean_allow_attention,
        )
        return create_block_mask(
            block_mask_function,
            B=batch_size,
            H=None,
            Q_LEN=query_length,
            KV_LEN=key_length,
            device=disallow_attention_flag.device,
        )

    additive_attention_mask = torch.zeros_like(
        disallow_attention_flag,
        dtype=torch.float32 if enable_mixed_precision else torch.bfloat16,
    )
    additive_attention_mask.masked_fill_(disallow_attention_flag, float("-inf"))
    return additive_attention_mask


def build_block_diffusion_doc_disallow_mask_from_doc_ids(
    document_ids: torch.Tensor,
    block_diffusion_disallow_mask: torch.Tensor,
) -> torch.Tensor:
    full_sequence_document_ids = torch.cat([document_ids, document_ids], dim=1)
    cross_document_flag = full_sequence_document_ids.unsqueeze(2) != full_sequence_document_ids.unsqueeze(1)
    return cross_document_flag.unsqueeze(1) | block_diffusion_disallow_mask.to(document_ids.device)
