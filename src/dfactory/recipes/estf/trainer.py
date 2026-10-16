"""Trainer for ESTF continued pretraining with m2t and t2t passes."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from dfactory.components.block_diffusion.attention import build_block_diffusion_attention_masks
from dfactory.components.block_diffusion.layout import build_paired_layout, interleave_paired_slots
from dfactory.data import TrainBatch
from dfactory.recipes.estf.algorithm import (
    initialize_random_known_mask,
    m2t_loss_weights,
    select_estf_tokens,
    token_confidence,
)
from dfactory.trainers import BaseTrainer, StepContext, StepMetrics

IGNORE_INDEX = -100


class ESTFTrainer(BaseTrainer):
    """Run per-block curriculum selection with paired m2t and t2t passes."""

    def training_step(self, batch: TrainBatch, context: StepContext) -> StepMetrics:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss_mask = batch["loss_mask"].bool()
        attention_spec = batch["attention_spec"]
        sequence_lengths = attention_spec["sequence_lengths"]
        valid_lengths = attention_spec["valid_lengths"]
        if input_ids.ndim != 2:
            raise ValueError("ESTF input_ids must have shape [batch, sequence]")

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
            clean_values=torch.full_like(labels, IGNORE_INDEX),
            fill_value=IGNORE_INDEX,
        )
        paired_trainable = interleave_paired_slots(
            loss_mask,
            sequence_lengths,
            clean_values=torch.zeros_like(loss_mask),
            fill_value=False,
        )
        eligible_mask = paired_trainable & layout.noisy_slot_mask & (paired_labels != IGNORE_INDEX)
        attention = build_block_diffusion_attention_masks(layout, backend=self.run_config.model.attention).base

        known_mask = torch.zeros_like(eligible_mask)
        if self.recipe_config.initialize_random_known:
            block_end_mask = layout.noisy_slot_mask & ((layout.position_ids + 1) % self.recipe_config.block_size == 0)
            known_mask = initialize_random_known_mask(
                eligible_mask,
                protected_mask=block_end_mask,
                ratio_range=self.recipe_config.initial_known_ratio,
                schedule_power=self.recipe_config.initial_known_schedule_power,
            )
        initial_known_count = known_mask.sum()
        phase_weight = 1.0 / (2 * self.recipe_config.max_stages)
        weighted_loss = paired_input_ids.new_zeros((), dtype=torch.float32)
        stage_metrics: dict[str, float] = {}
        total_eligible = eligible_mask.sum()
        final_covered = known_mask

        for stage in range(self.recipe_config.max_stages):
            current_mask = eligible_mask & ~known_mask
            m2t_input_ids = torch.where(
                current_mask,
                torch.full_like(paired_input_ids, self.special_tokens["mask_token_id"]),
                paired_input_ids,
            )
            m2t_outputs = context.forward(self._model_inputs(m2t_input_ids, layout.position_ids, attention))
            m2t_predictions = m2t_outputs.logits.detach().argmax(dim=-1)

            selection = None
            if stage < self.recipe_config.max_stages - 1:
                confidence = token_confidence(
                    m2t_outputs.logits,
                    paired_labels,
                    source=self.recipe_config.confidence_source,
                )
                selection = select_estf_tokens(
                    confidence,
                    eligible_mask=eligible_mask,
                    known_mask=known_mask,
                    group_ids=layout.noise_group_ids,
                    stage=stage,
                    num_stages=self.recipe_config.max_stages,
                    confidence_threshold=self.recipe_config.confidence_threshold,
                )
                m2t_loss_mask = selection.selected if self.recipe_config.selective_loss else current_mask
            else:
                m2t_loss_mask = current_mask

            if self.recipe_config.selective_loss:
                m2t_weights = m2t_loss_mask.float()
            else:
                m2t_weights = m2t_loss_weights(
                    current_mask,
                    eligible_mask=eligible_mask,
                    group_ids=layout.noise_group_ids,
                    block_size=self.recipe_config.block_size,
                )
            m2t_loss = self._weighted_objective(m2t_outputs.logits, paired_labels, m2t_weights)
            m2t_loss = context.reduce_loss(m2t_loss, m2t_loss_mask.sum())
            context.backward(m2t_loss, weight=phase_weight)
            weighted_loss = weighted_loss + m2t_loss.detach().float() * phase_weight

            if selection is not None:
                known_mask = known_mask | selection.selected
                confidence_count = selection.confidence_selected.sum()
                forced_count = selection.forced_fill.sum()
                new_count = selection.selected.sum()
            else:
                confidence_count = total_eligible.new_zeros(())
                forced_count = total_eligible.new_zeros(())
                new_count = current_mask.sum()

            t2t_known = known_mask if stage < self.recipe_config.max_stages - 1 else eligible_mask
            t2t_remaining = eligible_mask & ~t2t_known
            t2t_input_ids = torch.where(t2t_known, m2t_predictions, paired_input_ids)
            t2t_input_ids = torch.where(
                t2t_remaining,
                torch.full_like(t2t_input_ids, self.special_tokens["mask_token_id"]),
                t2t_input_ids,
            )
            t2t_outputs = context.forward(self._model_inputs(t2t_input_ids, layout.position_ids, attention))
            t2t_loss = self._weighted_objective(t2t_outputs.logits, paired_labels, t2t_known.float())
            t2t_loss = context.reduce_loss(t2t_loss, t2t_known.sum())
            context.backward(t2t_loss, weight=phase_weight)
            weighted_loss = weighted_loss + t2t_loss.detach().float() * phase_weight
            final_covered = t2t_known

            t2t_predictions = t2t_outputs.logits.detach().argmax(dim=-1)
            t2t_count = t2t_known.sum()
            if torch.any(t2t_known):
                t2t_error = ((t2t_predictions != paired_labels) & t2t_known).sum().float() / t2t_count.float()
            else:
                t2t_error = t2t_outputs.logits.new_zeros(())
            stage_metrics.update(
                {
                    f"stage_{stage}/m2t_loss": float(m2t_loss.detach().item()),
                    f"stage_{stage}/m2t_tokens": float(m2t_loss_mask.sum().detach().item()),
                    f"stage_{stage}/t2t_loss": float(t2t_loss.detach().item()),
                    f"stage_{stage}/t2t_tokens": float(t2t_count.detach().item()),
                    f"stage_{stage}/new_tokens": float(new_count.detach().item()),
                    f"stage_{stage}/confidence_tokens": float(confidence_count.detach().item()),
                    f"stage_{stage}/forced_fill_tokens": float(forced_count.detach().item()),
                    f"stage_{stage}/t2t_error_rate": float(t2t_error.detach().item()),
                }
            )

        covered_tokens = final_covered.sum()
        return StepMetrics(
            values={
                "loss": float(weighted_loss.item()),
                "loss_tokens": float(total_eligible.detach().item()),
                "initial_known_tokens": float(initial_known_count.detach().item()),
                "coverage": float((covered_tokens.float() / total_eligible.clamp_min(1).float()).detach().item()),
                **stage_metrics,
            },
            weight=max(float(total_eligible.detach().item()), 1.0),
        )

    @staticmethod
    def _model_inputs(input_ids: torch.Tensor, position_ids: torch.Tensor, attention_mask):
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _weighted_objective(
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        per_token = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).view_as(labels)
        return (per_token * weights.to(per_token.dtype)).mean()


__all__ = ["ESTFTrainer"]
