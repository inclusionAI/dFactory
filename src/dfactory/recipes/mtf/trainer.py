"""Multi-stage teacher-forcing trainer for the MTF recipe."""

from __future__ import annotations

import torch

from dfactory.components.block_diffusion import (
    apply_block_noise,
    build_block_diffusion_attention_masks,
    build_paired_layout,
    interleave_paired_slots,
    same_token_cross_entropy,
    select_curriculum_tokens,
)
from dfactory.data import TrainBatch
from dfactory.trainers import BaseTrainer, StepContext, StepMetrics


class MTFTrainer(BaseTrainer):
    """Run confidence-based teacher forcing over progressively larger token sets."""

    def training_step(self, batch: TrainBatch, context: StepContext) -> StepMetrics:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss_mask = batch["loss_mask"].bool()
        attention_spec = batch["attention_spec"]
        sequence_lengths = attention_spec["sequence_lengths"]
        valid_lengths = attention_spec["valid_lengths"]
        if input_ids.ndim != 2:
            raise ValueError("MTF input_ids must have shape [batch, sequence]")

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
            noise_group_ids=layout.noise_group_ids,
            mask_token_id=self.special_tokens["mask_token_id"],
            eos_token_id=self.special_tokens["eos_token_id"],
            noise_range=(1.0, 1.0),
            force_mask_token_ids=self.recipe_config.force_mask_token_ids,
        )
        attention_masks = build_block_diffusion_attention_masks(
            layout,
            backend=self.run_config.model.attention,
            sliding_window=self.recipe_config.sliding_window,
        )

        eligible_mask = noise.labels != -100
        selected_history = torch.zeros_like(eligible_mask)
        prediction_history = noise.input_ids.clone()
        stage_input_ids = noise.input_ids
        stage_losses: list[torch.Tensor] = []
        stage_metrics: dict[str, float] = {}
        total_eligible = eligible_mask.sum()

        for stage in range(self.recipe_config.max_stages):
            model_inputs = {
                "input_ids": stage_input_ids,
                "position_ids": layout.position_ids,
                "attention_mask": attention_masks.base,
            }
            if self.recipe_config.sliding_window is not None:
                model_inputs["block_diffusion_attention_masks"] = attention_masks
                model_inputs["training_step"] = context.global_step
            outputs = context.forward(model_inputs)
            predicted_ids = outputs.logits.detach().argmax(dim=-1)
            selection = select_curriculum_tokens(
                outputs.logits,
                eligible_mask=eligible_mask,
                selected_history=selected_history,
                group_ids=layout.noise_group_ids,
                stage=stage,
                num_stages=self.recipe_config.max_stages,
                confidence_threshold=self.recipe_config.confidence_threshold,
            )

            stage_labels = torch.full_like(noise.labels, -100)
            stage_labels[selection] = noise.labels[selection]
            objective = same_token_cross_entropy(outputs.logits, stage_labels)
            stage_loss = context.reduce_loss(objective.loss, objective.token_count)
            stage_losses.append(stage_loss)

            selected_count = selection.sum()
            prediction_history = torch.where(selection, predicted_ids, prediction_history)
            if torch.any(selection):
                error_rate = ((prediction_history != noise.labels) & selection).sum().float() / selected_count.float()
            else:
                error_rate = outputs.logits.new_zeros(())
            selected_history = selected_history | selection
            stage_input_ids = torch.where(selected_history, noise.labels, noise.input_ids)

            stage_metrics[f"stage_{stage}/loss"] = float(stage_loss.detach().item())
            stage_metrics[f"stage_{stage}/tokens"] = float(selected_count.detach().item())
            stage_metrics[f"stage_{stage}/selection_ratio"] = float(
                (selected_count.float() / total_eligible.clamp_min(1).float()).detach().item()
            )
            stage_metrics[f"stage_{stage}/error_rate"] = float(error_rate.detach().item())

        total_loss = torch.stack(stage_losses).mean()
        context.backward(total_loss)
        covered_tokens = selected_history.sum()
        values = {
            "loss": float(total_loss.detach().item()),
            "loss_tokens": float(total_eligible.detach().item()),
            "coverage": float((covered_tokens.float() / total_eligible.clamp_min(1).float()).detach().item()),
            **stage_metrics,
        }
        return StepMetrics(
            values=values,
            weight=max(float(total_eligible.detach().item()), 1.0),
        )


__all__ = ["MTFTrainer"]
