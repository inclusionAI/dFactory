"""Trainer for the EMTF teacher-forcing and self-correcting recipe."""

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


class EMTFTrainer(BaseTrainer):
    """Train each curriculum stage with TF followed by self-correction."""

    def training_step(self, batch: TrainBatch, context: StepContext) -> StepMetrics:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss_mask = batch["loss_mask"].bool()
        attention_spec = batch["attention_spec"]
        sequence_lengths = attention_spec["sequence_lengths"]
        valid_lengths = attention_spec["valid_lengths"]
        if input_ids.ndim != 2:
            raise ValueError("EMTF input_ids must have shape [batch, sequence]")

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
        teacher_forcing_input_ids = noise.input_ids
        phase_weight = 1.0 / (2 * self.recipe_config.max_stages)
        tf_losses: list[torch.Tensor] = []
        sc_losses: list[torch.Tensor] = []
        stage_metrics: dict[str, float] = {}
        total_eligible = eligible_mask.sum()

        for stage in range(self.recipe_config.max_stages):
            tf_outputs = context.forward(
                self._model_inputs(
                    teacher_forcing_input_ids,
                    layout.position_ids,
                    attention_masks,
                    context,
                )
            )
            tf_predictions = tf_outputs.logits.detach().argmax(dim=-1)
            selection = select_curriculum_tokens(
                tf_outputs.logits,
                eligible_mask=eligible_mask,
                selected_history=selected_history,
                group_ids=layout.noise_group_ids,
                stage=stage,
                num_stages=self.recipe_config.max_stages,
                confidence_threshold=self.recipe_config.confidence_threshold,
            )
            tf_objective = self._objective(tf_outputs.logits, noise.labels, selection)
            tf_loss = context.reduce_loss(tf_objective.loss, tf_objective.token_count)
            context.backward(tf_loss, weight=phase_weight)
            tf_losses.append(tf_loss.detach())

            sc_mask = selected_history | selection
            prediction_history = torch.where(sc_mask, tf_predictions, prediction_history)
            sc_input_ids = torch.where(sc_mask, prediction_history, noise.input_ids)
            sc_outputs = context.forward(
                self._model_inputs(
                    sc_input_ids,
                    layout.position_ids,
                    attention_masks,
                    context,
                )
            )
            sc_predictions = sc_outputs.logits.detach().argmax(dim=-1)
            sc_objective = self._objective(sc_outputs.logits, noise.labels, sc_mask)
            sc_loss = context.reduce_loss(sc_objective.loss, sc_objective.token_count)
            context.backward(sc_loss, weight=phase_weight)
            sc_losses.append(sc_loss.detach())

            selected_count = selection.sum()
            sc_count = sc_mask.sum()
            if torch.any(selection):
                tf_error_rate = (
                    (prediction_history != noise.labels) & selection
                ).sum().float() / selected_count.float()
            else:
                tf_error_rate = tf_outputs.logits.new_zeros(())
            if torch.any(sc_mask):
                sc_error_rate = ((sc_predictions != noise.labels) & sc_mask).sum().float() / sc_count.float()
            else:
                sc_error_rate = sc_outputs.logits.new_zeros(())

            selected_history = sc_mask
            teacher_forcing_input_ids = torch.where(selected_history, noise.labels, noise.input_ids)
            stage_metrics.update(
                {
                    f"stage_{stage}/tf_loss": float(tf_loss.detach().item()),
                    f"stage_{stage}/tf_tokens": float(selected_count.detach().item()),
                    f"stage_{stage}/sc_loss": float(sc_loss.detach().item()),
                    f"stage_{stage}/sc_tokens": float(sc_count.detach().item()),
                    f"stage_{stage}/selection_ratio": float(
                        (selected_count.float() / total_eligible.clamp_min(1).float()).detach().item()
                    ),
                    f"stage_{stage}/error_rate": float(tf_error_rate.detach().item()),
                    f"stage_{stage}/sc_error_rate": float(sc_error_rate.detach().item()),
                }
            )

        tf_loss_total = torch.stack(tf_losses).sum() * phase_weight
        sc_loss_total = torch.stack(sc_losses).sum() * phase_weight
        total_loss = tf_loss_total + sc_loss_total
        covered_tokens = selected_history.sum()
        return StepMetrics(
            values={
                "loss": float(total_loss.item()),
                "tf_loss": float(tf_loss_total.item()),
                "sc_loss": float(sc_loss_total.item()),
                "loss_tokens": float(total_eligible.detach().item()),
                "coverage": float((covered_tokens.float() / total_eligible.clamp_min(1).float()).detach().item()),
                **stage_metrics,
            },
            weight=max(float(total_eligible.detach().item()), 1.0),
        )

    def _model_inputs(self, input_ids, position_ids, attention_masks, context):
        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_masks.base,
        }
        if self.recipe_config.sliding_window is not None:
            model_inputs["block_diffusion_attention_masks"] = attention_masks
            model_inputs["training_step"] = context.global_step
        return model_inputs

    @staticmethod
    def _objective(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        selected_labels = torch.full_like(labels, -100)
        selected_labels[mask] = labels[mask]
        return same_token_cross_entropy(logits, selected_labels)


__all__ = ["EMTFTrainer"]
