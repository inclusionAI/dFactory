"""Conversation-to-token transformation for SFT recipes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from dfactory.data.batch import TrainBatch

IGNORE_INDEX = -100


def _message_value(record: dict[str, Any], text_key: str) -> list[dict[str, Any]]:
    messages = record.get(text_key)
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"record field {text_key!r} must be a non-empty list of messages")
    for message in messages:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise ValueError("every message must contain role and content")
    return messages


def _native_assistant_mask(
    tokenizer: Any, messages: list[dict[str, Any]], max_seq_len: int
) -> tuple[list[int], list[int]] | None:
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=max_seq_len,
        )
    except (TypeError, ValueError):
        return None
    input_ids = encoded.get("input_ids")
    mask = encoded.get("assistant_masks") or encoded.get("assistant_tokens_mask")
    if input_ids is None or mask is None or len(input_ids) != len(mask):
        return None
    return list(input_ids), list(mask)


def _fallback_assistant_mask(
    tokenizer: Any, messages: list[dict[str, Any]], max_seq_len: int
) -> tuple[list[int], list[int]]:
    input_ids = list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
        )
    )
    mask = [0] * len(input_ids)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = messages[:index]
        start_ids = tokenizer.apply_chat_template(
            prefix,
            tokenize=True,
            add_special_tokens=False,
            add_generation_prompt=True,
        )
        end_ids = tokenizer.apply_chat_template(
            messages[: index + 1],
            tokenize=True,
            add_special_tokens=False,
            add_generation_prompt=False,
        )
        start = min(len(start_ids), len(mask))
        end = min(len(end_ids), len(mask))
        mask[start:end] = [1] * max(0, end - start)
    return input_ids, mask


def transform_conversation(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    max_seq_len: int,
    block_size: int,
    pad_token_id: int,
    text_key: str = "messages",
    pad_to_max_length: bool = True,
) -> TrainBatch:
    """Tokenize a conversation and label every assistant span without adding noise."""
    if max_seq_len <= 0 or max_seq_len % block_size:
        raise ValueError("max_seq_len must be positive and divisible by block_size")
    messages = _message_value(record, text_key)
    encoded = _native_assistant_mask(tokenizer, messages, max_seq_len)
    input_ids, assistant_mask = encoded or _fallback_assistant_mask(tokenizer, messages, max_seq_len)
    input_ids = input_ids[:max_seq_len]
    assistant_mask = assistant_mask[: len(input_ids)]
    valid_length = len(input_ids)
    target_length = max_seq_len if pad_to_max_length else ((valid_length + block_size - 1) // block_size) * block_size
    pad_length = target_length - valid_length

    ids = torch.tensor(input_ids + [pad_token_id] * pad_length, dtype=torch.long)
    trainable = torch.tensor(assistant_mask + [0] * pad_length, dtype=torch.bool)
    labels = ids.clone()
    labels[~trainable] = IGNORE_INDEX
    return {
        "input_ids": ids,
        "labels": labels,
        "loss_mask": trainable,
        "position_ids": torch.arange(target_length, dtype=torch.long),
        "attention_spec": {
            "sequence_lengths": torch.tensor([target_length], dtype=torch.long),
            "valid_lengths": torch.tensor([valid_length], dtype=torch.long),
        },
    }


class TransformDataset(torch.utils.data.Dataset[TrainBatch]):
    def __init__(self, source: Sequence[dict[str, Any]], transform: Any) -> None:
        self.source = source
        self.transform = transform

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> TrainBatch:
        return self.transform(self.source[index])


__all__ = ["IGNORE_INDEX", "TransformDataset", "transform_conversation"]
