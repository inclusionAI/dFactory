"""Block-internal ordering objective for SolDiff pretraining."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _extract_token_scores(model_output):
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "token_scores"):
        return model_output.token_scores
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, (tuple, list)) and model_output:
        return model_output[0]
    raise TypeError(f"Cannot extract token scores from {type(model_output)!r}.")


def _squeeze_token_scores_to_2d(
    token_scores: torch.Tensor,
    expected_shape: tuple[int, int],
    name: str,
) -> torch.Tensor:
    if token_scores.shape == expected_shape:
        return token_scores
    if token_scores.shape == (*expected_shape, 1):
        return token_scores.squeeze(-1)
    raise ValueError(f"{name} token_scores shape mismatch: expected {expected_shape}, got {tuple(token_scores.shape)}")


def module_zero_loss_anchor(module: torch.nn.Module, device) -> torch.Tensor:
    for parameter in module.parameters():
        if parameter.requires_grad:
            return parameter.sum()
    return torch.zeros((), device=device, dtype=torch.float32)


def build_local_grouped_tensors(
    document_ids: torch.Tensor,
    valid_token_mask: torch.Tensor,
    block_size: int,
) -> dict[str, torch.Tensor]:
    device = document_ids.device
    batch_size, sequence_length = document_ids.shape

    flat_valid_token_mask = valid_token_mask.reshape(-1)
    flat_valid_token_indices = torch.nonzero(flat_valid_token_mask, as_tuple=False).squeeze(-1)

    if flat_valid_token_indices.numel() == 0:
        return {
            "has_active_groups": torch.tensor(False, device=device),
        }

    token_positions = torch.arange(sequence_length, device=device).unsqueeze(0).expand(batch_size, -1)
    physical_block_ids = torch.div(token_positions, block_size, rounding_mode="floor")
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, sequence_length)

    flat_batch_indices = batch_indices.reshape(-1)[flat_valid_token_mask]
    flat_block_indices = physical_block_ids.reshape(-1)[flat_valid_token_mask]
    flat_document_indices = document_ids.reshape(-1)[flat_valid_token_mask]

    total_block_count = (sequence_length + block_size - 1) // block_size
    document_stride = sequence_length + 1
    batch_stride = total_block_count * document_stride

    flat_group_keys = flat_batch_indices * batch_stride + flat_block_indices * document_stride + flat_document_indices
    unique_group_keys, inverse_group_indices = torch.unique(flat_group_keys, sorted=True, return_inverse=True)

    token_ordering_stride = batch_size * sequence_length
    token_ordering_keys = inverse_group_indices.to(torch.long) * token_ordering_stride + flat_valid_token_indices
    _unused_sorted_keys, sorted_token_order = torch.sort(token_ordering_keys)
    sorted_group_indices = inverse_group_indices[sorted_token_order]
    sorted_token_indices = flat_valid_token_indices[sorted_token_order]

    group_token_counts = torch.bincount(sorted_group_indices, minlength=unique_group_keys.shape[0])
    max_group_token_count = int(group_token_counts.max().item())
    if max_group_token_count > block_size:
        raise RuntimeError(
            "Detected local-order group length larger than block_size, "
            "which violates (physical_block_id, document_id) grouping."
        )

    sorted_token_positions = torch.arange(sorted_group_indices.shape[0], device=device)
    group_start_offsets = torch.cumsum(group_token_counts, dim=0) - group_token_counts
    slot_index_in_group = sorted_token_positions - group_start_offsets[sorted_group_indices]

    dense_group_token_indices = torch.full(
        (unique_group_keys.shape[0], max_group_token_count),
        -1,
        dtype=torch.long,
        device=device,
    )
    dense_group_token_indices[sorted_group_indices, slot_index_in_group] = sorted_token_indices
    slot_positions = torch.arange(max_group_token_count, device=device).unsqueeze(0)
    valid_group_slot_mask = slot_positions < group_token_counts.unsqueeze(1)

    group_batch_indices = torch.div(unique_group_keys, batch_stride, rounding_mode="floor")

    return {
        "has_active_groups": torch.tensor(True, device=device),
        "dense_group_token_indices": dense_group_token_indices,
        "group_token_counts": group_token_counts,
        "valid_group_slot_mask": valid_group_slot_mask,
        "group_batch_indices": group_batch_indices,
    }


def sample_two_local_orders_from_qlogits(
    dense_group_qlogits: torch.Tensor,
    dense_group_token_indices: torch.Tensor,
    group_token_counts: torch.Tensor,
    group_batch_indices: torch.Tensor,
    shared_mask_ratio: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = dense_group_qlogits.device
    group_count, max_group_token_count = dense_group_qlogits.shape
    if group_count == 0:
        return {
            "has_active_groups": torch.tensor(False, device=device),
        }

    slot_positions = torch.arange(max_group_token_count, device=device).unsqueeze(0)
    valid_group_slot_mask = slot_positions < group_token_counts.unsqueeze(1)
    dense_group_qlogits = dense_group_qlogits.masked_fill(~valid_group_slot_mask, float("-inf"))

    with torch.no_grad():
        gumbel_noise = (
            -torch.empty(
                (2, group_count, max_group_token_count),
                device=device,
                dtype=torch.float32,
            )
            .exponential_()
            .log()
        )
        sampled_scores = dense_group_qlogits.detach().float().unsqueeze(0) + gumbel_noise
        sampled_slot_orders = torch.argsort(sampled_scores, dim=-1, descending=True)

    sampled_group_token_orders = torch.gather(
        dense_group_token_indices.unsqueeze(0).expand(2, -1, -1),
        dim=2,
        index=sampled_slot_orders,
    )

    group_mask_ratio = shared_mask_ratio[group_batch_indices]
    masked_token_counts = torch.ceil(group_mask_ratio * group_token_counts.float()).to(torch.long)
    masked_token_counts = masked_token_counts.clamp(min=1)
    masked_token_counts = torch.minimum(masked_token_counts, group_token_counts)
    visible_token_counts = group_token_counts - masked_token_counts

    remaining_slot_mask = (slot_positions >= visible_token_counts.unsqueeze(1)) & (
        slot_positions < group_token_counts.unsqueeze(1)
    )

    return {
        "has_active_groups": torch.tensor(True, device=device),
        "dense_group_qlogits": dense_group_qlogits,
        "sampled_slot_orders": sampled_slot_orders,
        "sampled_group_token_orders": sampled_group_token_orders,
        "group_token_counts": group_token_counts,
        "visible_token_counts": visible_token_counts,
        "remaining_slot_mask": remaining_slot_mask,
        "group_batch_indices": group_batch_indices,
    }


def build_two_masked_inputs_from_orders(
    clean_input_ids: torch.Tensor,
    sampled_group_token_orders: torch.Tensor,
    remaining_slot_mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    two_path_masked_input_ids = clean_input_ids.unsqueeze(0).expand(2, -1, -1).clone()

    for sampled_path_index in range(2):
        token_indices_to_mask = sampled_group_token_orders[sampled_path_index][remaining_slot_mask]
        two_path_masked_input_ids[sampled_path_index].reshape(-1)[token_indices_to_mask] = mask_token_id
    return two_path_masked_input_ids


def compute_active_cross_entropy(
    masked_noisy_logits: torch.Tensor,
    active_combined_token_indices: torch.Tensor,
    active_target_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute active-token CE with a stable, vocabulary-chunked normalizer."""
    total_active_tokens = active_combined_token_indices.shape[0]
    if total_active_tokens == 0:
        empty_output = torch.empty((0,), device=masked_noisy_logits.device, dtype=torch.float32)
        return empty_output

    sequence_length = masked_noisy_logits.shape[1]
    vocab_size = masked_noisy_logits.shape[-1]
    vocab_chunk_size = 4096
    tiny_value = torch.finfo(torch.float32).tiny
    active_row_indices = torch.div(
        active_combined_token_indices,
        sequence_length,
        rounding_mode="floor",
    )
    active_col_indices = active_combined_token_indices % sequence_length

    running_max = torch.full(
        (total_active_tokens,),
        float("-inf"),
        device=masked_noisy_logits.device,
        dtype=torch.float32,
    )
    running_exp_sum = torch.zeros(
        (total_active_tokens,),
        device=masked_noisy_logits.device,
        dtype=torch.float32,
    )
    for vocab_start in range(0, vocab_size, vocab_chunk_size):
        vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
        logits_slice = masked_noisy_logits[
            active_row_indices,
            active_col_indices,
            vocab_start:vocab_end,
        ].float()
        slice_max = logits_slice.max(dim=-1).values
        updated_max = torch.maximum(running_max, slice_max)
        previous_scale = torch.exp(running_max - updated_max)
        slice_exp = torch.exp(logits_slice - updated_max.unsqueeze(-1))
        running_exp_sum = previous_scale * running_exp_sum + slice_exp.sum(dim=-1)
        running_max = updated_max

        del logits_slice, slice_max, updated_max, previous_scale, slice_exp

    active_logsumexp = running_max + torch.log(running_exp_sum.clamp_min(tiny_value))
    active_target_logits = masked_noisy_logits[
        active_row_indices,
        active_col_indices,
        active_target_token_ids,
    ].float()
    active_cross_entropy = active_logsumexp - active_target_logits

    del (
        active_row_indices,
        active_col_indices,
        running_max,
        running_exp_sum,
        active_logsumexp,
        active_target_logits,
    )
    return active_cross_entropy


def compute_detached_logit_moments_for_metrics(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    finite_logits = logits[valid_mask].detach().float()
    if finite_logits.numel() > 0:
        logits_sum_for_metrics = finite_logits.sum()
        logits_sq_sum_for_metrics = (finite_logits * finite_logits).sum()
        logits_count_for_metrics = torch.tensor(
            float(finite_logits.numel()),
            device=logits.device,
            dtype=torch.float32,
        )
        logits_abs_max_for_metrics = finite_logits.abs().max()
    else:
        logits_sum_for_metrics = torch.zeros((), device=logits.device, dtype=torch.float32)
        logits_sq_sum_for_metrics = torch.zeros((), device=logits.device, dtype=torch.float32)
        logits_count_for_metrics = torch.zeros((), device=logits.device, dtype=torch.float32)
        logits_abs_max_for_metrics = torch.zeros((), device=logits.device, dtype=torch.float32)
    return (
        logits_sum_for_metrics,
        logits_sq_sum_for_metrics,
        logits_count_for_metrics,
        logits_abs_max_for_metrics,
    )


def compute_order_metrics_from_sampled_orders(
    natural_tokens: torch.Tensor,
    natural_valid_mask: torch.Tensor,
    sampled_group_token_orders: torch.Tensor,
    group_token_counts: torch.Tensor,
    visible_token_counts: torch.Tensor,
    block_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return rank diagnostics and the remaining set used by the objective."""
    device = natural_tokens.device
    num_paths, _group_count, max_group_token_count = natural_tokens.shape

    step_ids = torch.arange(max_group_token_count, device=device)
    prefix_visible_mask = (step_ids.view(1, 1, -1) < visible_token_counts.view(1, -1, 1)).expand(num_paths, -1, -1)
    prefix_token_valid_mask = prefix_visible_mask & (sampled_group_token_orders >= 0)
    prefix_selected_mask = (
        (natural_tokens.unsqueeze(-1) == sampled_group_token_orders.unsqueeze(-2))
        & prefix_token_valid_mask.unsqueeze(-2)
    ).any(dim=-1)
    remaining_natural_mask = natural_valid_mask & (~prefix_selected_mask)

    decoded_order_valid_mask = sampled_group_token_orders >= 0
    natural_to_decoded_match = (
        (natural_tokens.unsqueeze(-1) == sampled_group_token_orders.unsqueeze(-2))
        & natural_valid_mask.unsqueeze(-1)
        & decoded_order_valid_mask.unsqueeze(-2)
    )
    decoded_match_count_by_natural = natural_to_decoded_match.sum(dim=-1)
    if not torch.all(decoded_match_count_by_natural[natural_valid_mask] == 1):
        raise RuntimeError(
            "Decoded order mapping is not one-to-one: each valid natural position must map to exactly one decoded rank."
        )
    decoded_match_count_by_decoded = natural_to_decoded_match.sum(dim=-2)
    if not torch.all(decoded_match_count_by_decoded[decoded_order_valid_mask] == 1):
        raise RuntimeError(
            "Decoded order mapping is not one-to-one: each valid decoded rank must map to exactly one natural position."
        )
    decoded_rank_axis = torch.arange(max_group_token_count, device=device, dtype=torch.float32).view(1, 1, 1, -1)
    decoded_rank_by_natural_position = (natural_to_decoded_match.to(torch.float32) * decoded_rank_axis).sum(dim=-1)
    decoded_rank_by_natural_position = decoded_rank_by_natural_position.masked_fill(~natural_valid_mask, 0.0)

    rank_normalizer = float(max(block_size - 1, 1))
    normalized_rank_by_position = decoded_rank_by_natural_position / rank_normalizer
    avg_rank_by_position_sum = torch.zeros((block_size,), device=device, dtype=torch.float32)
    avg_rank_by_position_count = torch.zeros((block_size,), device=device, dtype=torch.float32)
    local_rank_position_sum = (normalized_rank_by_position * natural_valid_mask.to(torch.float32)).sum(dim=(0, 1))
    local_rank_position_count = natural_valid_mask.to(torch.float32).sum(dim=(0, 1))
    avg_rank_by_position_sum[:max_group_token_count] = local_rank_position_sum
    avg_rank_by_position_count[:max_group_token_count] = local_rank_position_count

    pairwise_valid_mask = (
        natural_valid_mask.unsqueeze(-1)
        & natural_valid_mask.unsqueeze(-2)
        & (
            torch.arange(max_group_token_count, device=device).view(1, 1, -1, 1)
            < torch.arange(max_group_token_count, device=device).view(1, 1, 1, -1)
        )
    )
    inversion_count = (
        (
            (decoded_rank_by_natural_position.unsqueeze(-1) > decoded_rank_by_natural_position.unsqueeze(-2))
            & pairwise_valid_mask
        )
        .to(torch.float32)
        .sum(dim=(-1, -2))
    )
    group_lengths = group_token_counts.view(1, -1).expand(num_paths, -1).to(torch.float32)
    max_inversion_count = (group_lengths * (group_lengths - 1.0) * 0.5).clamp_min(1.0)
    inversion_norm = torch.where(
        group_lengths > 1.0,
        inversion_count / max_inversion_count,
        torch.zeros_like(inversion_count),
    )
    inversion_norm_sum = inversion_norm.sum()

    remaining_count_from_mask = remaining_natural_mask.to(torch.long).sum(dim=-1)
    expected_remaining_count = (group_token_counts - visible_token_counts).view(1, -1).expand(num_paths, -1)
    if not torch.equal(remaining_count_from_mask, expected_remaining_count):
        raise RuntimeError(
            "Remaining-mask cardinality mismatch: decoded prefix does not produce the expected remaining set size."
        )

    metric_eps = 1e-6
    if ((inversion_norm < -metric_eps) | (inversion_norm > 1.0 + metric_eps)).any():
        raise RuntimeError("Normalized inversion is out of range [0, 1]. Check residual decode and normalization.")

    return (
        avg_rank_by_position_sum,
        avg_rank_by_position_count,
        inversion_norm_sum,
        remaining_natural_mask,
        expected_remaining_count,
    )


def compute_prior_permutation_alignment_loss(
    dense_group_prior_logits: torch.Tensor,
    sampled_slot_orders: torch.Tensor,
    group_token_counts: torch.Tensor,
    visible_token_counts: torch.Tensor,
    group_batch_indices: torch.Tensor,
    batch_size: int,
    tau: float,
) -> torch.Tensor:
    num_paths, group_count, max_group_token_count = dense_group_prior_logits.shape
    if num_paths != 2:
        raise RuntimeError("Prior permutation alignment loss expects exactly two sampled orders.")
    if group_count == 0:
        return dense_group_prior_logits.sum() * 0.0
    if tau <= 0:
        raise ValueError("suffix_reconstruction_tau must be positive.")

    device = dense_group_prior_logits.device
    step_ids = torch.arange(max_group_token_count, device=device)
    valid_step_mask = step_ids.view(1, -1) < group_token_counts.view(-1, 1)
    suffix_step_mask = valid_step_mask.unsqueeze(0) & (
        step_ids.view(1, 1, -1) >= visible_token_counts.view(1, group_count, 1)
    )
    suffix_token_counts = group_token_counts - visible_token_counts
    decision_token_counts = (suffix_token_counts - 1).clamp_min(0)
    decision_step_mask = suffix_step_mask & (step_ids.view(1, 1, -1) < (group_token_counts - 1).view(1, group_count, 1))

    prior_logits_by_sampled_order = torch.gather(
        dense_group_prior_logits.float(),
        dim=2,
        index=sampled_slot_orders,
    )
    # The Plackett-Luce likelihood temperature remains fixed at 1.0; tau only
    # controls the rank-decay weights below.
    suffix_prior_logits = prior_logits_by_sampled_order.masked_fill(~suffix_step_mask, float("-inf"))
    suffix_log_denominator = torch.logcumsumexp(suffix_prior_logits.flip(-1), dim=-1).flip(-1)
    suffix_token_nll = (suffix_log_denominator - prior_logits_by_sampled_order).masked_fill(~suffix_step_mask, 0.0)
    suffix_ranks = (step_ids.view(1, 1, -1) - visible_token_counts.view(1, group_count, 1)).to(torch.float32)
    decision_weights = torch.exp(-suffix_ranks / float(tau)).masked_fill(~decision_step_mask, 0.0)
    decision_weights = (
        decision_weights / decision_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    ).expand(num_paths, -1, -1)
    per_path_group_loss = decision_token_counts.to(suffix_token_nll.dtype).view(1, group_count) * (
        decision_weights * suffix_token_nll
    ).sum(dim=-1)
    per_group_alignment_loss = 0.5 * (per_path_group_loss[0] + per_path_group_loss[1])

    per_batch_alignment_loss = torch.zeros(
        (batch_size,),
        dtype=per_group_alignment_loss.dtype,
        device=device,
    )
    per_batch_alignment_loss.scatter_add_(0, group_batch_indices, per_group_alignment_loss)
    return per_batch_alignment_loss.mean()


def compute_suffix_reconstruction_loss(
    active_cross_entropy: torch.Tensor,
    active_path_indices: torch.Tensor,
    active_group_indices: torch.Tensor,
    active_candidate_slot_indices: torch.Tensor,
    sampled_slot_orders: torch.Tensor,
    group_token_counts: torch.Tensor,
    visible_token_counts: torch.Tensor,
    group_batch_indices: torch.Tensor,
    group_count: int,
    batch_size: int,
    tau: float,
) -> torch.Tensor:
    if group_count == 0 or active_cross_entropy.numel() == 0:
        return active_cross_entropy.sum() * 0.0
    if tau <= 0:
        raise ValueError("suffix_reconstruction_tau must be positive.")

    num_paths, sampled_group_count, max_group_token_count = sampled_slot_orders.shape
    if num_paths != 2:
        raise RuntimeError("Suffix reconstruction loss expects exactly two sampled orders.")
    if sampled_group_count != group_count:
        raise RuntimeError("Suffix reconstruction loss group count mismatch between active indices and sampled orders.")

    device = active_cross_entropy.device
    sampled_step_ids = torch.arange(max_group_token_count, device=device)
    suffix_step_mask = (sampled_step_ids.view(1, 1, -1) >= visible_token_counts.view(1, group_count, 1)) & (
        sampled_step_ids.view(1, 1, -1) < group_token_counts.view(1, group_count, 1)
    )
    suffix_ranks = (sampled_step_ids.view(1, 1, -1) - visible_token_counts.view(1, group_count, 1)).to(torch.float32)
    suffix_weights_by_sampled_order = torch.exp(-suffix_ranks / float(tau)).masked_fill(~suffix_step_mask, 0.0)
    suffix_weights_by_sampled_order = (
        (
            suffix_weights_by_sampled_order
            / suffix_weights_by_sampled_order.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        )
        .expand(num_paths, -1, -1)
        .to(active_cross_entropy.dtype)
    )

    suffix_weights_by_natural_slot = torch.zeros(
        (2, group_count, max_group_token_count),
        dtype=active_cross_entropy.dtype,
        device=device,
    )
    suffix_weights_by_natural_slot.scatter_(2, sampled_slot_orders, suffix_weights_by_sampled_order)
    active_suffix_weights = suffix_weights_by_natural_slot[
        active_path_indices, active_group_indices, active_candidate_slot_indices
    ]

    active_segment_indices = active_path_indices * group_count + active_group_indices
    segment_count = 2 * group_count
    per_segment_reconstruction_loss = torch.zeros(
        (segment_count,),
        dtype=active_cross_entropy.dtype,
        device=active_cross_entropy.device,
    )
    per_segment_reconstruction_loss.scatter_add_(
        0, active_segment_indices, active_suffix_weights * active_cross_entropy
    )
    per_path_group_reconstruction_loss = per_segment_reconstruction_loss.view(2, group_count)
    suffix_token_counts = (group_token_counts - visible_token_counts).to(per_path_group_reconstruction_loss.dtype)
    per_path_group_reconstruction_loss = suffix_token_counts.view(1, group_count) * per_path_group_reconstruction_loss
    per_group_reconstruction_loss = 0.5 * (
        per_path_group_reconstruction_loss[0] + per_path_group_reconstruction_loss[1]
    )

    per_batch_reconstruction_loss = torch.zeros(
        (batch_size,),
        dtype=per_group_reconstruction_loss.dtype,
        device=active_cross_entropy.device,
    )
    per_batch_reconstruction_loss.scatter_add_(0, group_batch_indices, per_group_reconstruction_loss)
    return per_batch_reconstruction_loss.mean()


def compute_local_residual_rloo_loss(
    dense_group_qlogits: torch.Tensor,
    dense_group_prior_logits: torch.Tensor,
    sampled_slot_orders: torch.Tensor,
    sampled_group_token_orders: torch.Tensor,
    dense_group_token_indices: torch.Tensor,
    group_token_counts: torch.Tensor,
    visible_token_counts: torch.Tensor,
    remaining_slot_mask: torch.Tensor,
    group_batch_indices: torch.Tensor,
    masked_noisy_logits: torch.Tensor,
    clean_input_ids: torch.Tensor,
    block_size: int,
    suffix_reconstruction_tau: float,
    lambda_align: float,
    lambda_recon: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    group_count, max_group_token_count = dense_group_qlogits.shape
    batch_size, sequence_length = clean_input_ids.shape

    step_ids = torch.arange(max_group_token_count, device=clean_input_ids.device)
    valid_step_mask = step_ids.unsqueeze(0) < group_token_counts.unsqueeze(1)
    valid_step_mask_by_path = valid_step_mask.unsqueeze(0).expand(2, -1, -1)

    (
        q_logits_sum_for_metrics,
        q_logits_sq_sum_for_metrics,
        q_logits_count_for_metrics,
        q_logits_abs_max_for_metrics,
    ) = compute_detached_logit_moments_for_metrics(
        logits=dense_group_qlogits,
        valid_mask=valid_step_mask,
    )

    qlogits_by_sampled_order = torch.gather(
        dense_group_qlogits.unsqueeze(0).expand(2, -1, -1),
        dim=2,
        index=sampled_slot_orders,
    )  # Reorder q logits by each group's sampled token order.

    qlogits_by_sampled_order = qlogits_by_sampled_order.masked_fill(~valid_step_mask_by_path, float("-inf"))
    reversed_suffix_lse = torch.logcumsumexp(qlogits_by_sampled_order.flip(-1), dim=-1).flip(-1)
    prefix_step_logprob = (qlogits_by_sampled_order - reversed_suffix_lse).masked_fill(~valid_step_mask_by_path, 0.0)
    prefix_cumsum = torch.cumsum(prefix_step_logprob, dim=-1)
    gather_prefix_index = (visible_token_counts - 1).clamp(min=0).view(1, group_count, 1).expand(2, -1, -1)
    gathered_prefix_logprob = torch.gather(prefix_cumsum, 2, gather_prefix_index).squeeze(-1)
    prefix_logprob = torch.where(
        visible_token_counts.view(1, -1) > 0,
        gathered_prefix_logprob,
        torch.zeros_like(gathered_prefix_logprob),
    )

    remaining_slot_mask_by_path = remaining_slot_mask.unsqueeze(0).expand(2, -1, -1)

    current_candidate_counts = (group_token_counts - visible_token_counts).unsqueeze(0).expand(2, -1)
    rank_ids = torch.arange(block_size, device=clean_input_ids.device)
    valid_candidate_rank_mask = rank_ids.view(1, 1, -1) < current_candidate_counts.unsqueeze(-1)

    natural_tokens = dense_group_token_indices.unsqueeze(0).expand(2, -1, -1)
    natural_valid_mask = valid_step_mask_by_path
    (
        avg_rank_by_position_sum,
        avg_rank_by_position_count,
        inversion_norm_sum,
        remaining_natural_mask,
        expected_remaining_count,
    ) = compute_order_metrics_from_sampled_orders(
        natural_tokens=natural_tokens,
        natural_valid_mask=natural_valid_mask,
        sampled_group_token_orders=sampled_group_token_orders,
        group_token_counts=group_token_counts,
        visible_token_counts=visible_token_counts,
        block_size=block_size,
    )

    natural_slot_ids = (
        torch.arange(max_group_token_count, device=clean_input_ids.device).view(1, 1, -1).expand(2, group_count, -1)
    )
    masked_slot_ids = torch.where(
        remaining_natural_mask,
        natural_slot_ids,
        torch.full_like(natural_slot_ids, max_group_token_count),
    )
    sorted_remaining_slot_ids = torch.sort(masked_slot_ids, dim=-1).values
    safe_sorted_remaining_slot_ids = sorted_remaining_slot_ids.clamp(max=max_group_token_count - 1)
    candidate_token_indices_local = torch.gather(
        natural_tokens,
        dim=2,
        index=safe_sorted_remaining_slot_ids,
    )
    candidate_token_indices = torch.full(
        (2, group_count, block_size),
        -1,
        dtype=torch.long,
        device=clean_input_ids.device,
    )
    candidate_token_indices[:, :, :max_group_token_count] = candidate_token_indices_local
    candidate_token_indices = candidate_token_indices.masked_fill(~valid_candidate_rank_mask, -1)

    active_path_indices, active_group_indices, active_rank_indices = torch.where(valid_candidate_rank_mask)
    active_original_token_indices = candidate_token_indices[
        active_path_indices, active_group_indices, active_rank_indices
    ]
    if (active_original_token_indices < 0).any():
        raise RuntimeError("Found invalid candidate token index during natural-order residual-rank decode.")
    active_combined_token_indices = active_original_token_indices + active_path_indices * (batch_size * sequence_length)

    # For q_k in F, use remaining-set posterior over tokens, and map rank k by natural-order
    # position among remaining tokens.
    natural_qlogits = dense_group_qlogits.unsqueeze(0).expand(2, -1, -1).float()
    remaining_natural_qlogits = natural_qlogits.masked_fill(~remaining_natural_mask, float("-inf"))
    remaining_natural_qlogsumexp = torch.logsumexp(remaining_natural_qlogits, dim=-1)
    active_candidate_slot_indices = safe_sorted_remaining_slot_ids[
        active_path_indices, active_group_indices, active_rank_indices
    ]
    active_qstep_logprob = (
        natural_qlogits[active_path_indices, active_group_indices, active_candidate_slot_indices]
        - remaining_natural_qlogsumexp[active_path_indices, active_group_indices]
    )
    active_qstep_probabilities = torch.exp(active_qstep_logprob)

    remaining_count_from_suffix_mask = remaining_slot_mask_by_path.to(torch.long).sum(dim=-1)
    if not torch.equal(remaining_count_from_suffix_mask, expected_remaining_count):
        raise RuntimeError(
            "Remaining-slot suffix cardinality mismatch: "
            "sampled-order suffix does not match expected remaining set size."
        )

    active_target_token_ids = clean_input_ids.reshape(-1)[active_original_token_indices]
    active_cross_entropy = compute_active_cross_entropy(
        masked_noisy_logits=masked_noisy_logits,
        active_combined_token_indices=active_combined_token_indices,
        active_target_token_ids=active_target_token_ids,
    )
    suffix_reconstruction_loss = compute_suffix_reconstruction_loss(
        active_cross_entropy=active_cross_entropy,
        active_path_indices=active_path_indices,
        active_group_indices=active_group_indices,
        active_candidate_slot_indices=active_candidate_slot_indices,
        sampled_slot_orders=sampled_slot_orders,
        group_token_counts=group_token_counts,
        visible_token_counts=visible_token_counts,
        group_batch_indices=group_batch_indices,
        group_count=group_count,
        batch_size=batch_size,
        tau=suffix_reconstruction_tau,
    )

    prior_logits_by_natural_slot = dense_group_prior_logits.masked_fill(~valid_step_mask_by_path, float("-inf"))
    prior_logits_by_rank_local = torch.gather(
        prior_logits_by_natural_slot,
        dim=2,
        index=safe_sorted_remaining_slot_ids,
    )
    prior_logits_by_rank = torch.full(
        (2, group_count, block_size),
        float("-inf"),
        device=clean_input_ids.device,
        dtype=torch.float32,
    )
    prior_logits_by_rank[:, :, :max_group_token_count] = prior_logits_by_rank_local
    prior_logits_by_rank = prior_logits_by_rank.masked_fill(~valid_candidate_rank_mask, float("-inf"))
    safe_prior_logits = torch.where(
        valid_candidate_rank_mask,
        prior_logits_by_rank,
        torch.full_like(prior_logits_by_rank, -1e30),
    )
    prior_logprobs = F.log_softmax(safe_prior_logits, dim=-1).masked_fill(~valid_candidate_rank_mask, float("-inf"))
    (
        prior_logits_sum_for_metrics,
        prior_logits_sq_sum_for_metrics,
        prior_logits_count_for_metrics,
        prior_logits_abs_max_for_metrics,
    ) = compute_detached_logit_moments_for_metrics(
        logits=prior_logits_by_rank,
        valid_mask=valid_candidate_rank_mask,
    )

    active_prior_logprob = prior_logprobs[active_path_indices, active_group_indices, active_rank_indices]
    active_f_values = active_qstep_probabilities * (active_prior_logprob - active_cross_entropy - active_qstep_logprob)
    active_segment_indices = active_path_indices * group_count + active_group_indices
    segment_count = 2 * group_count
    per_segment_f_values = torch.zeros(
        (segment_count,),
        dtype=active_f_values.dtype,
        device=active_f_values.device,
    )
    per_segment_f_values.scatter_add_(0, active_segment_indices, active_f_values)
    per_path_group_f_values = per_segment_f_values.view(2, group_count)

    per_group_score_term = (prefix_logprob[0] - prefix_logprob[1]) * (
        per_path_group_f_values[0] - per_path_group_f_values[1]
    ).detach()
    per_group_objective = (
        group_token_counts.to(per_path_group_f_values.dtype)
        * 0.5
        * (per_group_score_term + per_path_group_f_values[0] + per_path_group_f_values[1])
    )

    per_batch_objective = torch.zeros(
        (batch_size,),
        dtype=per_group_objective.dtype,
        device=per_group_objective.device,
    )
    per_batch_objective.scatter_add_(0, group_batch_indices, per_group_objective)

    local_rloo_loss = -per_batch_objective.mean()
    prior_permutation_alignment_loss = compute_prior_permutation_alignment_loss(
        dense_group_prior_logits=dense_group_prior_logits,
        sampled_slot_orders=sampled_slot_orders,
        group_token_counts=group_token_counts,
        visible_token_counts=visible_token_counts,
        group_batch_indices=group_batch_indices,
        batch_size=batch_size,
        tau=suffix_reconstruction_tau,
    )
    loss = (
        local_rloo_loss
        + float(lambda_align) * prior_permutation_alignment_loss
        + float(lambda_recon) * suffix_reconstruction_loss
    )
    ce_sum_for_metrics = active_cross_entropy.detach().sum()
    active_weighted_ce_values = active_qstep_probabilities * active_cross_entropy
    weighted_ce_sum_for_metrics = active_weighted_ce_values.detach().sum()
    active_token_count_for_metrics = torch.tensor(
        float(active_cross_entropy.numel()),
        device=clean_input_ids.device,
        dtype=torch.float32,
    )
    active_group_count_for_metrics = torch.tensor(
        float(2 * group_count),
        device=clean_input_ids.device,
        dtype=torch.float32,
    )

    return (
        loss,
        local_rloo_loss,
        prior_permutation_alignment_loss,
        suffix_reconstruction_loss,
        ce_sum_for_metrics,
        weighted_ce_sum_for_metrics,
        active_token_count_for_metrics,
        active_group_count_for_metrics,
        q_logits_sum_for_metrics,
        q_logits_sq_sum_for_metrics,
        q_logits_count_for_metrics,
        q_logits_abs_max_for_metrics,
        prior_logits_sum_for_metrics,
        prior_logits_sq_sum_for_metrics,
        prior_logits_count_for_metrics,
        prior_logits_abs_max_for_metrics,
        avg_rank_by_position_sum,
        avg_rank_by_position_count,
        inversion_norm_sum,
    )
