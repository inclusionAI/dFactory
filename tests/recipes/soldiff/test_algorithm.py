"""SolDiff attention rules, objectives and gradient regressions."""

import pytest
import torch

from dfactory.recipes.soldiff.algorithm import (
    build_local_grouped_tensors,
    build_two_masked_inputs_from_orders,
    compute_active_cross_entropy,
    compute_local_residual_rloo_loss,
    compute_order_metrics_from_sampled_orders,
    compute_prior_permutation_alignment_loss,
    compute_suffix_reconstruction_loss,
    sample_two_local_orders_from_qlogits,
)
from dfactory.recipes.soldiff.attention import (
    block_diffusion_mask,
    build_attention_mask_by_impl,
    build_block_diffusion_doc_disallow_mask_from_doc_ids,
    build_clean_blockwise_doc_attn_mask,
    build_doc_ids_from_eos,
)


def _reference_case():
    clean = torch.tensor([[1, 2, 9, 3, 4, 5], [9, 2, 3, 4, 9, 6]])
    groups = build_local_grouped_tensors(build_doc_ids_from_eos(clean, 9), torch.ones_like(clean, dtype=torch.bool), 4)
    q = torch.linspace(-1.1, 1.2, groups["dense_group_token_indices"].numel())
    q = q.reshape_as(groups["dense_group_token_indices"]).requires_grad_()
    masked_q = q.masked_fill(~groups["valid_group_slot_mask"], float("-inf"))
    torch.manual_seed(1729)
    sampled = sample_two_local_orders_from_qlogits(
        masked_q,
        groups["dense_group_token_indices"],
        groups["group_token_counts"],
        groups["group_batch_indices"],
        torch.tensor([0.5, 0.9]),
    )
    torch.manual_seed(248)
    prior = torch.randn((2,) + tuple(q.shape), requires_grad=True)
    logits = torch.randn((4, 6, 13), requires_grad=True)
    return clean, groups, q, masked_q, sampled, prior, logits


def test_groups_respect_both_eos_and_physical_blocks() -> None:
    clean, groups, *_ = _reference_case()
    assert groups["dense_group_token_indices"].tolist() == [
        [0, 1, 2],
        [3, -1, -1],
        [4, 5, -1],
        [6, -1, -1],
        [7, 8, 9],
        [10, -1, -1],
        [11, -1, -1],
    ]
    assert groups["group_token_counts"].tolist() == [3, 1, 2, 1, 3, 1, 1]
    assert groups["group_batch_indices"].tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert int(groups["group_token_counts"].sum()) == clean.numel()


def test_two_pl_samples_and_masking_match_fixed_rng_reference() -> None:
    clean, groups, _, _, sampled, *_ = _reference_case()
    assert sampled["sampled_slot_orders"].tolist() == [
        [[2, 1, 0], [0, 1, 2], [1, 0, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
        [[1, 0, 2], [0, 1, 2], [1, 0, 2], [0, 1, 2], [2, 0, 1], [0, 1, 2], [0, 1, 2]],
    ]
    # ceil(mask_ratio * group_size), shared between both paths, always masks at least one token.
    assert sampled["visible_token_counts"].tolist() == [1, 0, 1, 0, 0, 0, 0]
    masked = build_two_masked_inputs_from_orders(
        clean, sampled["sampled_group_token_orders"], sampled["remaining_slot_mask"], 12
    )
    assert masked.shape == (2, 2, 6)
    assert (masked == 12).sum(dim=(1, 2)).tolist() == [10, 10]
    for path in range(2):
        for group, count in enumerate(groups["group_token_counts"]):
            visible = int(sampled["visible_token_counts"][group])
            order = sampled["sampled_group_token_orders"][path, group, :count]
            for token in order[:visible]:
                assert masked[path].reshape(-1)[token] == clean.reshape(-1)[token]
            assert (masked[path].reshape(-1)[order[visible:]] == 12).all()


def test_full_objective_and_all_three_gradients_match_fixed_reference() -> None:
    clean, groups, q, masked_q, sampled, prior, logits = _reference_case()
    objective = compute_local_residual_rloo_loss(
        masked_q,
        prior,
        sampled["sampled_slot_orders"],
        sampled["sampled_group_token_orders"],
        groups["dense_group_token_indices"],
        groups["group_token_counts"],
        sampled["visible_token_counts"],
        sampled["remaining_slot_mask"],
        groups["group_batch_indices"],
        logits,
        clean,
        4,
        8.0,
        1.0,
        1.0,
    )
    assert [float(value.detach()) for value in objective[:8]] == pytest.approx(
        [41.637462616, 21.748060226, 3.086794853, 16.802608490, 67.379287720, 46.276344299, 20.0, 14.0],
        rel=2e-6,
        abs=2e-6,
    )
    gradients = torch.autograd.grad(objective[0], (q, prior, logits))
    assert [float(value.abs().sum()) for value in gradients] == pytest.approx(
        [4.159137726, 3.821018696, 20.710098267], rel=2e-6, abs=2e-6
    )
    torch.testing.assert_close(
        gradients[0],
        torch.tensor(
            [
                [0.972440004, 0.727257252, -1.699697256],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-0.086108446, -0.293763041, 0.379871368],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        rtol=2e-6,
        atol=2e-6,
    )
    assert len(objective) == 19
    assert objective[16].tolist() == pytest.approx([2.0, 4 / 3, 4 / 3, 0.0])
    assert objective[17].tolist() == [14.0, 6.0, 4.0, 0.0]
    assert float(objective[18]) == pytest.approx(4.0)
    assert all(torch.isfinite(value).all() for value in gradients)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab_size", [17, 4096, 5003])
def test_chunked_ce_and_gradients_equal_dense_log_softmax(dtype, vocab_size) -> None:
    # Cross the 4096-token vocabulary chunk boundary and stress the stable running maximum.
    torch.manual_seed(100)
    logits = (torch.randn(2, 3, vocab_size) * 20).to(dtype).requires_grad_()
    indices = torch.tensor([0, 2, 4, 5])
    targets = torch.tensor([0, min(4095, vocab_size - 1), min(4096, vocab_size - 1), vocab_size - 1])
    ce = compute_active_cross_entropy(logits, indices, targets)
    active = logits.reshape(-1, vocab_size)[indices].float()
    logprob = active.log_softmax(-1)
    expected_ce = -logprob[torch.arange(4), targets]
    torch.testing.assert_close(ce, expected_ce, atol=2e-5, rtol=2e-6)
    actual_grad = torch.autograd.grad(ce.sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected_ce.sum(), logits)[0]
    # BF16 cast/scatter backward can round before cancellation at the target.
    gradient_atol = torch.finfo(dtype).eps if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual_grad, expected_grad, atol=gradient_atol, rtol=2e-4)


def test_ce_handles_no_active_tokens() -> None:
    result = compute_active_cross_entropy(
        torch.zeros(2, 3, 17, dtype=torch.bfloat16),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )
    assert result.shape == (0,)
    assert result.dtype == torch.float32


def test_order_diagnostics_preserve_the_remaining_set_used_by_the_objective() -> None:
    natural = torch.tensor([[[0, 1, 2]], [[0, 1, 2]]])
    result = compute_order_metrics_from_sampled_orders(
        natural,
        torch.ones_like(natural, dtype=torch.bool),
        torch.tensor([[[2, 0, 1]], [[1, 2, 0]]]),
        torch.tensor([3]),
        torch.tensor([1]),
        4,
    )
    assert len(result) == 5
    assert result[3].tolist() == [[[True, True, False]], [[True, False, True]]]
    assert result[4].tolist() == [[2], [2]]


@pytest.mark.parametrize(
    ("orders", "counts", "message"),
    [
        ([[[2, 2, 1]], [[1, 2, 0]]], [3], "not one-to-one"),
        ([[[2, 0, 1]], [[1, 2, 0]]], [2], "Remaining-mask cardinality mismatch"),
    ],
)
def test_order_diagnostics_retain_mapping_and_remaining_set_checks(orders, counts, message) -> None:
    natural = torch.tensor([[[0, 1, 2]], [[0, 1, 2]]])
    with pytest.raises(RuntimeError, match=message):
        compute_order_metrics_from_sampled_orders(
            natural,
            torch.ones_like(natural, dtype=torch.bool),
            torch.tensor(orders),
            torch.tensor(counts),
            torch.tensor([1]),
            4,
        )


def test_suffix_losses_use_rank_decay_and_exclude_final_pl_decision() -> None:
    prior = torch.tensor([[[2.0, 0.0, -1.0]], [[2.0, 0.0, -1.0]]], requires_grad=True)
    orders = torch.tensor([[[0, 2, 1]], [[0, 2, 1]]])
    counts = torch.tensor([3])
    visible = torch.tensor([1])
    batch = torch.tensor([0])
    align = compute_prior_permutation_alignment_loss(prior, orders, counts, visible, batch, 1, 8.0)
    # The two-token suffix has exactly one nontrivial PL decision: choose token 2 over token 1.
    torch.testing.assert_close(align, torch.log1p(torch.exp(torch.tensor(1.0))))
    ce = torch.tensor([5.0, 7.0, 5.0, 7.0], requires_grad=True)
    recon = compute_suffix_reconstruction_loss(
        ce,
        torch.tensor([0, 0, 1, 1]),
        torch.zeros(4, dtype=torch.long),
        torch.tensor([2, 1, 2, 1]),
        orders,
        counts,
        visible,
        batch,
        1,
        1,
        8.0,
    )
    weights = torch.tensor([0.0, -1.0 / 8]).softmax(0)
    torch.testing.assert_close(recon, 2 * (weights * torch.tensor([5.0, 7.0])).sum())
    recon.backward()
    torch.testing.assert_close(ce.grad, weights.repeat(2))


def test_eos_belongs_to_previous_document_and_clean_attention_is_block_causal() -> None:
    tokens = torch.tensor([[1, 9, 2, 3, 9, 4]])
    disallow, document_ids = build_clean_blockwise_doc_attn_mask(tokens, 2, 9)
    assert document_ids.tolist() == [[0, 0, 1, 1, 1, 2]]
    for query in range(6):
        for key in range(6):
            expected = document_ids[0, query] == document_ids[0, key] and query // 2 >= key // 2
            assert bool(~disallow[0, 0, query, key]) == bool(expected)
    assert not disallow[0, 0, 2, 3]  # Both directions are visible within the same block.
    assert disallow[0, 0, 4, 5]  # Same physical block still cannot cross an EOS boundary.


def test_paired_attention_preserves_teacher_forcing_without_same_block_clean_leakage() -> None:
    length, block_size = 6, 2
    indices = torch.arange(2 * length)
    allow = block_diffusion_mask(None, None, indices[:, None], indices[None, :], block_size, length)
    documents = build_doc_ids_from_eos(torch.tensor([[1, 9, 2, 3, 9, 4]]), 9)
    disallow = build_block_diffusion_doc_disallow_mask_from_doc_ids(documents, ~allow[None, None])
    full_documents = documents.repeat(1, 2)
    for query in range(2 * length):
        for key in range(2 * length):
            query_clean, key_clean = query >= length, key >= length
            query_block, key_block = (query % length) // block_size, (key % length) // block_size
            same_doc = full_documents[0, query] == full_documents[0, key]
            expected = same_doc and (
                (not query_clean and not key_clean and query_block == key_block)
                or (not query_clean and key_clean and query_block > key_block)
                or (query_clean and key_clean and query_block >= key_block)
            )
            assert bool(~disallow[0, 0, query, key]) == bool(expected)
    assert disallow[0, 0, 3, length + 3]  # Noisy block cannot see its clean answers.
    assert not disallow[0, 0, 4, length + 2]  # Earlier clean block in same document is visible.


def test_sdpa_and_flex_use_the_same_allowed_pairs() -> None:
    disallow, _ = build_clean_blockwise_doc_attn_mask(torch.tensor([[1, 9, 2, 3, 9, 4]]), 2, 9)
    additive = build_attention_mask_by_impl(disallow, "sdpa", True)
    assert additive.dtype == torch.float32
    assert torch.equal(additive == float("-inf"), disallow)
    assert (additive[~disallow] == 0).all()
    block_mask = build_attention_mask_by_impl(disallow, "flex_attention", True)
    for query in range(6):
        for key in range(6):
            allowed = block_mask.mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor(query), torch.tensor(key))
            assert bool(allowed) == bool(~disallow[0, 0, query, key])
