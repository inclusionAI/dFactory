"""Deterministic, unpadded token rows for the SolDiff pretraining method."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import torch

from dfactory.config import DataConfig, RunConfig
from dfactory.data import TrainBatch
from dfactory.data.collators import collate_train_batch
from dfactory.recipes import RecipeData
from dfactory.recipes.soldiff.config import SolDiffConfig


def _document_tokens(record: Mapping[str, Any], token_key: str) -> list[int]:
    if not isinstance(record, Mapping) or token_key not in record:
        raise ValueError(f"soldiff token record is missing field {token_key!r}")
    value = record[token_key]
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise ValueError(f"token record field {token_key!r} must be a one-dimensional integer sequence")
        values = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        values = list(value)
    else:
        raise ValueError(f"token record field {token_key!r} must be a one-dimensional integer sequence")
    if not values:
        raise ValueError(f"token record field {token_key!r} must not be empty")
    if any(not isinstance(token, Integral) or isinstance(token, bool) or token < 0 for token in values):
        raise ValueError(f"token record field {token_key!r} must contain non-negative integers only")
    return [int(token) for token in values]


class FixedTokenRows(torch.utils.data.Dataset[TrainBatch]):
    """Concatenate EOS-terminated records globally, then slice complete rows.

    A missing final EOS is inserted once per source record. Existing internal EOS
    tokens are preserved. Long documents continue into the next row; positions
    and physical block indices restart at each row. The final incomplete row is
    discarded, never padded. Every rank builds the same plan before the shared
    trainer's DistributedSampler partitions and shuffles complete rows.

    Only source offsets and lengths are retained. Source records are read again
    on access and must remain unchanged during a training run.
    """

    def __init__(
        self, source: Sequence[Mapping[str, Any]], *, max_seq_len: int, eos_token_id: int, token_key: str
    ) -> None:
        if not isinstance(max_seq_len, int) or isinstance(max_seq_len, bool) or max_seq_len <= 0:
            raise ValueError("data.max_seq_len must be a positive integer")
        if not isinstance(eos_token_id, Integral) or isinstance(eos_token_id, bool) or eos_token_id < 0:
            raise ValueError("soldiff requires a non-negative integer eos_token_id")
        self.source = source
        self.max_seq_len = max_seq_len
        self.eos_token_id = int(eos_token_id)
        self.token_key = token_key
        self._document_ends: list[int] = []
        self._raw_lengths: list[int] = []
        self._append_eos: list[bool] = []
        total = 0
        for index in range(len(source)):
            tokens = _document_tokens(source[index], token_key)
            append_eos = tokens[-1] != self.eos_token_id
            self._raw_lengths.append(len(tokens))
            self._append_eos.append(append_eos)
            total += len(tokens) + int(append_eos)
            self._document_ends.append(total)
        self.total_tokens = total
        self.inserted_eos_tokens = sum(self._append_eos)
        self.discarded_tail_tokens = total % max_seq_len
        if total < max_seq_len:
            raise ValueError(
                f"soldiff requires at least one complete row of {max_seq_len} tokens; "
                f"source has {total} tokens including document EOS tokens"
            )

    def __len__(self) -> int:
        return self.total_tokens // self.max_seq_len

    def __getitem__(self, index: int) -> TrainBatch:
        if not isinstance(index, Integral) or isinstance(index, bool) or not 0 <= index < len(self):
            raise IndexError(f"soldiff row index out of range: {index}")
        start = int(index) * self.max_seq_len
        stop = start + self.max_seq_len
        document_index = bisect_right(self._document_ends, start)
        tokens: list[int] = []
        while start < stop:
            document_start = 0 if document_index == 0 else self._document_ends[document_index - 1]
            document_stop = self._document_ends[document_index]
            document = _document_tokens(self.source[document_index], self.token_key)
            if len(document) != self._raw_lengths[document_index] or (
                (document[-1] != self.eos_token_id) != self._append_eos[document_index]
            ):
                raise ValueError("soldiff source document lengths or EOS boundaries changed after row planning")
            if self._append_eos[document_index]:
                document.append(self.eos_token_id)
            segment_stop = min(stop, document_stop)
            tokens.extend(document[start - document_start : segment_stop - document_start])
            start = segment_stop
            document_index += 1
        input_ids = torch.tensor(tokens, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "loss_mask": torch.ones(self.max_seq_len, dtype=torch.bool),
            "position_ids": torch.arange(self.max_seq_len, dtype=torch.long),
            "attention_spec": {
                "sequence_lengths": torch.tensor([self.max_seq_len], dtype=torch.long),
                "valid_lengths": torch.tensor([self.max_seq_len], dtype=torch.long),
            },
        }


def validate_run(*, recipe_config: SolDiffConfig, run_config: RunConfig) -> None:
    recipe_config.validate()
    if run_config.model.provider != "qwen3":
        raise ValueError("soldiff requires model.provider=qwen3")
    if run_config.data.format != "token_records":
        raise ValueError("soldiff supports data.format=token_records only")
    if run_config.data.packing:
        raise ValueError("soldiff requires data.packing=false; its fixed token rows must not be packed again")
    if (
        not isinstance(run_config.data.max_seq_len, int)
        or isinstance(run_config.data.max_seq_len, bool)
        or run_config.data.max_seq_len <= 0
    ):
        raise ValueError("data.max_seq_len must be a positive integer")
    if run_config.data.max_seq_len % recipe_config.block_size:
        raise ValueError("data.max_seq_len must be divisible by recipe.params.block_size")
    if run_config.distributed.ep != 1:
        raise ValueError("soldiff requires distributed.ep=1")
    from dfactory.recipes.soldiff.q_config import load_q_config

    load_q_config(recipe_config.q_config_path)
    if recipe_config.p_head_config_path and not Path(recipe_config.p_head_config_path).is_file():
        raise ValueError("recipe.params.p_head_config_path must identify an existing JSON file")


def build_data(
    *,
    source: Any,
    tokenizer: Any,
    data_config: DataConfig,
    recipe_config: SolDiffConfig,
    special_tokens: Mapping[str, int],
) -> RecipeData:
    del tokenizer
    recipe_config.validate()
    if data_config.format != "token_records":
        raise ValueError("soldiff supports data.format=token_records only")
    if data_config.packing:
        raise ValueError("soldiff requires data.packing=false; its fixed token rows must not be packed again")
    if (
        not isinstance(data_config.max_seq_len, int)
        or isinstance(data_config.max_seq_len, bool)
        or data_config.max_seq_len <= 0
    ):
        raise ValueError("data.max_seq_len must be a positive integer")
    if data_config.max_seq_len % recipe_config.block_size:
        raise ValueError("data.max_seq_len must be divisible by recipe.params.block_size")
    dataset = FixedTokenRows(
        source,
        max_seq_len=data_config.max_seq_len,
        eos_token_id=special_tokens["eos_token_id"],
        token_key=data_config.text_key,
    )
    return RecipeData(dataset=dataset, collate_fn=collate_train_batch)


__all__ = ["FixedTokenRows", "build_data", "validate_run"]
