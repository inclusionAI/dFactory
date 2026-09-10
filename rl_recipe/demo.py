"""
Run the complete one-sequence algorithm demonstration.
Core conrtibutors: Yihong Zhuang, Zenan Huang, Chengxi Li, Kaiyuan Guan
"""

from __future__ import annotations

from .lumen_zoo import DemoLumenZoo, RolloutRequest
from .packed_ebpo_loss import packed_ebpo_loss
from .stepmap_noise import add_stepmap_noise
from .pseudo_tokenizer import PseudoTokenizer


def main() -> None:
    tokenizer = PseudoTokenizer()

    # A complete, readable example rollout used throughout this demo.
    question = "Who wrote Pride and Prejudice, and when was it first published?"
    answer = "Jane Austen wrote Pride and Prejudice; it was first published in 1813."

    request = RolloutRequest(
        task_id="search-001",
        task_type="search",
        question=question,
        tools=("search",),
    )
    result = DemoLumenZoo(fixed_answer=answer).run(request)
    answer = result.final_answer
    prompt_ids = tokenizer.encode(question)
    answer_ids = tokenizer.encode(answer)
    # The completed example rollout is the clean sequence x_0 and remains the target.
    target_ids = prompt_ids + answer_ids

    print("1. Agentic request -> lumen-zoo rollout")
    print("   route: request -> lumen-zoo -> result")
    print(f"   question: {question}")
    print(f"   reward: {result.reward}")
    print(f"   final response: {answer}")

    print("\n2. Text -> tokens")
    print(f"   rollout target shape: [1, {len(target_ids)}]")

    # Fixed LowConfidence rollout steps. Larger step values mean that a token was
    # committed later. Each entry keeps [previous token id, reveal step],
    # matching the downstream data contract.
    reveal_steps = [1, 2, 4, 5, 6, 7, 1, 3, 4, 5, 6, 7, 2, 3]
    response_step_maps = [
        [token_id, reveal_step]
        for token_id, reveal_step in zip(answer_ids, reveal_steps)
    ]
    sglang_result = {
        "output_ids": answer_ids,
        "meta_info": {
            "step_maps": response_step_maps,
        },
    }
    print("\n3. LowConfidence rollout")
    print(f"   intent: commit confident tokens first; uncertain tokens receive later steps")
    print(f"   fixed response: {answer}")
    print(f"   reveal steps: {reveal_steps}")
    print(f"   SGLang fields: {list(sglang_result)} / {list(sglang_result['meta_info'])}")
    print(f"   simulated step_maps sample: {response_step_maps[:3]} ...")

    # The response contains generated-token entries; prompt positions use sentinels.
    simulated_packed_step_maps = [[-1, -1]] * len(prompt_ids) + response_step_maps
    perturbed_sequence, perturb_mask, thresholds = add_stepmap_noise(
        target_ids,
        prompt_length=len(prompt_ids),
        packed_step_maps=simulated_packed_step_maps,
        mask_token_id=tokenizer.mask_token_id,
        block_size=4,
    )
    print("\n4. all_block_random_step_maps")
    print("   intent: reconstruct an intermediate generation state block by block")
    print(f"   sampled block thresholds: {thresholds}")
    print(f"   target x_0: {' '.join(tokenizer.tokens(target_ids))}")
    print(f"   perturbed x_t: {' '.join(tokenizer.tokens(perturbed_sequence))}")
    print(f"   masked response tokens: {sum(perturb_mask)}")

    completion_mask = [False] * len(prompt_ids) + [True] * len(answer_ids)
    # In training, the current, rollout, and reference policies evaluate the same x_t.
    # Their logits are gathered at target_ids (the clean example rollout x_0).
    # These fixed values stand in for those three model forwards; no extra transform
    # is applied between model output and the loss.
    current_response_logprobs = [
        -0.08, -0.11, -0.17, -0.19, -0.21, -0.23, -0.07,
        -0.13, -0.16, -0.18, -0.20, -0.22, -0.10, -0.12,
    ]
    old_response_logprobs = [
        -0.10, -0.14, -0.21, -0.23, -0.25, -0.27, -0.09,
        -0.16, -0.20, -0.22, -0.24, -0.26, -0.13, -0.15,
    ]
    reference_response_logprobs = [
        -0.09, -0.12, -0.18, -0.20, -0.22, -0.24, -0.08,
        -0.14, -0.17, -0.19, -0.21, -0.23, -0.11, -0.13,
    ]
    prompt_logprobs = [0.0] * len(prompt_ids)
    current = prompt_logprobs + current_response_logprobs
    old = prompt_logprobs + old_response_logprobs
    reference = prompt_logprobs + reference_response_logprobs
    print("\n5. Model forward on noisy token IDs")
    print("   input: perturbed_sequence = x_t")
    print("   labels: target_ids = example rollout x_0")
    print(f"   gathered rollout-token log-probs shape: [1, {len(current)}]")

    loss = packed_ebpo_loss(
        current, old, reference, completion_mask, perturb_mask, advantage=0.8
    )
    print("\n6. PackedEBPOLoss (one sequence)")
    print("   intent: aggregate active tokens into ELBO, stabilize the policy update, then regularize")
    print(f"   current/old ELBO: {loss.current_elbo:.4f} / {loss.old_elbo:.4f}")
    print(f"   mean block ratio -> clipped: {loss.ratio:.4f} -> {loss.clipped_ratio:.4f}")
    print(f"   policy + weighted KL = total: {loss.total_loss:.4f}")


if __name__ == "__main__":
    main()
