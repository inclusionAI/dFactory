import torch

from dfactory.data.transforms.sft import transform_conversation


class MaskAwareTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        input_ids = list(range(1, len(messages) * 2 + 1))
        assistant_mask = []
        for message in messages:
            assistant_mask.extend([message["role"] == "assistant"] * 2)
        return {"input_ids": input_ids, "assistant_masks": assistant_mask}


def test_conversation_labels_every_assistant_span_without_precomputed_noise() -> None:
    example = transform_conversation(
        {
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ]
        },
        tokenizer=MaskAwareTokenizer(),
        max_seq_len=8,
        block_size=2,
        pad_token_id=0,
    )

    assert example["input_ids"].tolist() == list(range(1, 9))
    assert example["loss_mask"].tolist() == [False, False, True, True, False, False, True, True]
    assert example["labels"].tolist() == [-100, -100, 3, 4, -100, -100, 7, 8]
    assert "noisy_input_ids" not in example
    assert torch.equal(example["position_ids"], torch.arange(8))
    assert torch.equal(example["attention_spec"]["sequence_lengths"], torch.tensor([8]))
    assert torch.equal(example["attention_spec"]["valid_lengths"], torch.tensor([8]))
