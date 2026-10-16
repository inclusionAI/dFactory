"""Trainer for autoregressive-to-Block-Diffusion continued pretraining."""

from __future__ import annotations

import torch

from dfactory.components.block_diffusion.attention import build_block_diffusion_attention_masks
from dfactory.components.block_diffusion.layout import build_paired_layout, interleave_paired_slots
from dfactory.components.block_diffusion.noise import apply_block_noise
from dfactory.components.block_diffusion.objectives import row_mean_same_token_cross_entropy, shifted_cross_entropy
from dfactory.data import TrainBatch
from dfactory.trainers import BaseTrainer, StepContext, StepMetrics


class AR2BDTrainer(BaseTrainer):
    """Train clean token documents through a runtime-noised paired layout."""

    def training_step(self, batch: TrainBatch, context: StepContext) -> StepMetrics:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss_mask = batch["loss_mask"].bool()
        attention_spec = batch["attention_spec"]
        sequence_lengths = attention_spec["sequence_lengths"]
        valid_lengths = attention_spec["valid_lengths"]
        if input_ids.ndim != 2:
            raise ValueError("AR2BD input_ids must have shape [batch, sequence]")

        layout = build_paired_layout(
            sequence_lengths,
            block_size=self.recipe_config.block_size,
            total_length=input_ids.shape[1] * 2,
            valid_lengths=valid_lengths,
        )
        paired_input_ids = interleave_paired_slots(
            input_ids,
            sequence_lengths,
            fill_value=self.special_tokens.get("pad_token_id", 0),
        )
        paired_labels = interleave_paired_slots(
            labels,
            sequence_lengths,
            clean_values=torch.full_like(labels, -100),
            fill_value=-100,
        )
        paired_trainable = interleave_paired_slots(
            loss_mask,
            sequence_lengths,
            clean_values=torch.zeros_like(loss_mask),
            fill_value=False,
        )
        noise = apply_block_noise(
            paired_input_ids,
            paired_labels,
            trainable_mask=paired_trainable,
            noisy_slot_mask=layout.noisy_slot_mask,
            noise_group_ids=layout.pair_ids,
            mask_token_id=self.special_tokens["mask_token_id"],
            noise_range=self.recipe_config.noise_range,
        )
        attention_masks = build_block_diffusion_attention_masks(
            layout,
            backend=self.run_config.model.attention,
        )
        outputs = context.forward(
            {
                "input_ids": noise.input_ids,
                "position_ids": layout.position_ids,
                "attention_mask": attention_masks.base,
            }
        )

        if self.recipe_config.same_token_labels:
            objective = row_mean_same_token_cross_entropy(outputs.logits, noise.labels)
        else:
            shifted_labels = noise.labels.clone()
            document_starts = layout.noisy_slot_mask & (layout.position_ids == 0)
            shifted_labels[document_starts] = -100
            objective = shifted_cross_entropy(outputs.logits, shifted_labels)
        loss = context.reduce_loss(objective.loss, objective.token_count)
        context.backward(loss)
        token_count = float(objective.token_count.detach().item())
        return StepMetrics(
            values={"loss": float(loss.detach().item()), "loss_tokens": token_count},
            weight=max(token_count, 1.0),
        )


__all__ = ["AR2BDTrainer"]
