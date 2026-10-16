"""Greedy sequential packing for token examples."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from dfactory.data.batch import TrainBatch


class PackedDataset(torch.utils.data.Dataset[TrainBatch]):
    """Plan deterministic sequential packs without retaining transformed tensors."""

    def __init__(
        self,
        dataset: Sequence[TrainBatch],
        *,
        max_seq_len: int,
        pad_token_id: int,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self._dataset = dataset
        self._max_seq_len = max_seq_len
        self._pad_token_id = pad_token_id
        self._packs: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        current_length = 0
        for source_index in range(len(dataset)):
            example = dataset[source_index]
            length = int(example["attention_spec"]["sequence_lengths"].sum().item())
            if length <= 0 or length > max_seq_len:
                raise ValueError("packed examples must have a positive length within max_seq_len")
            if current and current_length + length > max_seq_len:
                self._packs.append(current)
                current = []
                current_length = 0
            current.append((source_index, length))
            current_length += length
        if current:
            self._packs.append(current)
        self._max_examples_per_pack = max((len(pack) for pack in self._packs), default=0)

    def __len__(self) -> int:
        return len(self._packs)

    def __getitem__(self, index: int) -> TrainBatch:
        entries = self._packs[index]
        examples = [self._dataset[source_index] for source_index, _ in entries]
        lengths = [length for _, length in entries]
        valid_lengths = [int(example["attention_spec"]["valid_lengths"].sum().item()) for example in examples]
        used = sum(lengths)
        padding = self._max_seq_len - used

        def combine(name: str, fill_value: int | bool) -> torch.Tensor:
            values = [example[name][:length] for example, length in zip(examples, lengths, strict=True)]
            result = torch.cat(values)
            if padding:
                result = torch.cat((result, torch.full((padding,), fill_value, dtype=result.dtype)))
            return result

        metadata_padding = self._max_examples_per_pack - len(lengths)
        return {
            "input_ids": combine("input_ids", self._pad_token_id),
            "labels": combine("labels", -100),
            "loss_mask": combine("loss_mask", False),
            "position_ids": combine("position_ids", 0),
            "attention_spec": {
                "sequence_lengths": torch.tensor(lengths + [0] * metadata_padding, dtype=torch.long),
                "valid_lengths": torch.tensor(valid_lengths + [0] * metadata_padding, dtype=torch.long),
            },
        }


__all__ = ["PackedDataset"]
