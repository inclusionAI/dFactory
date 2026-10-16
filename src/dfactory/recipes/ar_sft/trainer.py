"""Autoregressive supervised fine-tuning trainer for the AR-SFT recipe."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from dfactory.data import TrainBatch
from dfactory.recipes.ar_sft.attention import (
    build_attention_mask,
    build_document_layout,
    build_flex_attention_mask,
)
from dfactory.trainers import BaseTrainer, StepContext, StepMetrics


class ARSFTTrainer(BaseTrainer):
    """Train with autoregressive, document-isolated causal attention."""

    def training_step(self, batch: TrainBatch, context: StepContext) -> StepMetrics:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        if input_ids.ndim != 2:
            raise ValueError("AR SFT input_ids must have shape [batch, sequence]")

        attention_spec = batch["attention_spec"]
        layout = build_document_layout(
            attention_spec["sequence_lengths"],
            total_length=input_ids.shape[1],
            valid_lengths=attention_spec["valid_lengths"],
        )
        if self.run_config.model.attention == "flex_attention":
            attention_mask = build_flex_attention_mask(layout)
        else:
            allowed_attention = build_attention_mask(layout)
            attention_mask = torch.zeros(
                (*allowed_attention.shape[:1], 1, *allowed_attention.shape[1:]),
                dtype=torch.float32,
                device=allowed_attention.device,
            )
            attention_mask.masked_fill_(~allowed_attention.unsqueeze(1), float("-inf"))

        outputs = context.forward(
            {
                "input_ids": input_ids,
                "position_ids": layout.position_ids,
                "attention_mask": attention_mask,
            }
        )
        shifted_logits = outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        loss_mask = shifted_labels != -100
        token_count = loss_mask.sum()
        if torch.any(loss_mask):
            loss = functional.cross_entropy(shifted_logits[loss_mask], shifted_labels[loss_mask])
        else:
            loss = shifted_logits.sum() * 0.0

        loss = context.reduce_loss(loss, token_count)
        context.backward(loss)
        weight = max(float(token_count.detach().item()), 1.0)
        return StepMetrics(
            values={"loss": float(loss.detach().item()), "loss_tokens": float(token_count.detach().item())},
            weight=weight,
        )


__all__ = ["ARSFTTrainer"]
