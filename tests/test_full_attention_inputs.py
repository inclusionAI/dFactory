"""Full-attention training (`block_diffusion_mode: false`) must feed the model the masked sequence x_t."""

import sys
from pathlib import Path

import torch

# data_transform.py needs only torch. Importing the file directly skips tasks/dataset/__init__.py, which pulls in VeOmni.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks" / "dataset"))
from data_transform import full_attention_inputs, process_mdm_tokenized_example  # noqa: E402

MASK_TOKEN_ID = 156895  # the id both training scripts pass to the transforms
SEQ_LEN = 64
PROMPT_LEN = 16


def collated_micro_batch(batch_size=4, seed=0):
    """A micro batch as the dataloader stacks it: input_ids, noisy_input_ids, attention_mask, labels."""
    torch.manual_seed(seed)
    rows = []
    for _ in range(batch_size):
        example = {"input_ids": torch.randint(0, 1000, (SEQ_LEN,)).tolist(), "prompt_lengths": PROMPT_LEN}
        rows += process_mdm_tokenized_example(
            example, max_seq_len=SEQ_LEN, noise_range=(0.3, 0.8), mask_token_id=MASK_TOKEN_ID
        )
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def test_loss_positions_do_not_show_their_answer():
    micro_batch = collated_micro_batch()
    labels = micro_batch["labels"].clone()
    loss_positions = labels != -100
    assert loss_positions.any()

    inputs = full_attention_inputs(micro_batch)

    shown = int((inputs["input_ids"][loss_positions] == labels[loss_positions]).sum())
    assert shown == 0, f"input_ids shows the answer at {shown} of {int(loss_positions.sum())} loss positions"
    assert (inputs["input_ids"][loss_positions] == MASK_TOKEN_ID).all()


def test_input_ids_is_the_noisy_sequence():
    micro_batch = collated_micro_batch()
    noisy_input_ids = micro_batch["noisy_input_ids"].clone()

    inputs = full_attention_inputs(micro_batch)

    assert torch.equal(inputs["input_ids"], noisy_input_ids)
    # LLaDA2MoeModelLM.forward takes **kwargs, so a key left here would be dropped without an error.
    assert "noisy_input_ids" not in inputs
    assert inputs["attention_mask"] is None
