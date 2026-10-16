import torch

from dfactory.data.collators import collate_train_batch
from dfactory.data.packing import PackedDataset


def _example(values: list[int], valid_length: int) -> dict:
    length = len(values)
    ids = torch.tensor(values)
    trainable = torch.arange(length) < valid_length
    return {
        "input_ids": ids,
        "labels": ids.masked_fill(~trainable, -100),
        "loss_mask": trainable,
        "position_ids": torch.arange(length),
        "attention_spec": {
            "sequence_lengths": torch.tensor([length]),
            "valid_lengths": torch.tensor([valid_length]),
        },
    }


def test_packing_preserves_pair_boundaries_and_pads_metadata() -> None:
    dataset = PackedDataset(
        [_example([1, 2, 0, 0], 2), _example([3, 4, 5, 0], 3), _example([6] * 8, 8)],
        max_seq_len=8,
        pad_token_id=0,
    )

    assert len(dataset) == 2
    first = dataset[0]
    assert first["input_ids"].tolist() == [1, 2, 0, 0, 3, 4, 5, 0]
    assert first["position_ids"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert first["attention_spec"]["sequence_lengths"].tolist() == [4, 4]
    assert first["attention_spec"]["valid_lengths"].tolist() == [2, 3]


def test_packing_keeps_only_source_indices_and_reloads_examples() -> None:
    class CountingDataset:
        def __init__(self) -> None:
            self.reads: list[int] = []
            self.examples = [_example([1, 2, 0, 0], 2), _example([3, 4, 0, 0], 2)]

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            self.reads.append(index)
            return self.examples[index]

    source = CountingDataset()
    dataset = PackedDataset(source, max_seq_len=8, pad_token_id=0)

    assert source.reads == [0, 1]
    assert dataset._packs == [[(0, 4), (1, 4)]]

    dataset[0]

    assert source.reads == [0, 1, 0, 1]


def test_collator_preserves_the_public_attention_spec() -> None:
    batch = collate_train_batch([_example([1, 2, 0, 0], 2), _example([3, 4, 5, 0], 3)])

    assert batch["input_ids"].shape == (2, 4)
    assert batch["loss_mask"].shape == (2, 4)
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert batch["attention_spec"]["sequence_lengths"].tolist() == [[4], [4]]
    assert batch["attention_spec"]["valid_lengths"].tolist() == [[2], [3]]
