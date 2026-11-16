"""Two-path block-internal RLOO with a trainable posterior and attached prior."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from dfactory.data import TrainBatch
from dfactory.trainers import BaseTrainer, StepContext, StepMetrics

from .algorithm import (
    _extract_token_scores,
    _squeeze_token_scores_to_2d,
    build_local_grouped_tensors,
    build_two_masked_inputs_from_orders,
    compute_local_residual_rloo_loss,
    module_zero_loss_anchor,
    sample_two_local_orders_from_qlogits,
)
from .attention import (
    block_diffusion_mask,
    build_attention_mask_by_impl,
    build_block_diffusion_doc_disallow_mask_from_doc_ids,
    build_clean_blockwise_doc_attn_mask,
)

_RESULT_STATISTICS = (
    "ce_sum",
    "weighted_ce_sum",
    "active_token_count",
    "group_count",
    "q_logits_sum",
    "q_logits_sq_sum",
    "q_logits_count",
    "q_logits_abs_max",
    "prior_logits_sum",
    "prior_logits_sq_sum",
    "prior_logits_count",
    "prior_logits_abs_max",
    "avg_rank_by_position_sum",
    "avg_rank_by_position_count",
    "inversion_norm_sum",
)
_LOSS_NAMES = ("loss", "local_rloo_loss", "prior_permutation_alignment_loss", "suffix_reconstruction_loss")
_MAX_STATISTICS = frozenset(("q_logits_abs_max", "prior_logits_abs_max"))
_FLOAT_STATISTICS = frozenset(("ce_sum", "weighted_ce_sum", "active_token_count", "group_count"))


def _parameter_norm(module: torch.nn.Module) -> float:
    squared_norm = None
    for parameter in module.parameters():
        value = parameter.detach().float().pow(2).sum()
        squared_norm = value if squared_norm is None else squared_norm + value
    if squared_norm is None:
        return 0.0
    if hasattr(squared_norm, "full_tensor"):
        squared_norm = squared_norm.full_tensor()
    return float(squared_norm.sqrt().item())


class SolDiffTrainer(BaseTrainer):
    """Model preparation, training objective and metrics for SolDiff."""

    def _stdout_metric_sink(self, step: int, values: dict[str, float]) -> None:
        from .logging import log_step_metrics

        log_step_metrics(
            step,
            values,
            output_dir=self.run_config.run.output_dir,
            block_size=self.recipe_config.block_size,
            diagnostics_every_steps=self.recipe_config.diagnostics_every_steps,
        )

    def data_loader_generator(self) -> torch.Generator:
        # Recreating an iterator on resume must not advance the model's CPU RNG.
        # This recipe's document transforms are deterministic; row shuffling is
        # independently seeded by DistributedSampler and its epoch.
        return torch.Generator().manual_seed(self.run_config.run.seed)

    def prepare_main_model(self, model: Any, tokenizer: Any) -> Any:
        from .artifacts import prepare_p_head, validate_p_head_resume_config

        del tokenizer
        resolved = prepare_p_head(
            model,
            model_path=self.run_config.model.weights_path,
            p_head_config_path=self.recipe_config.p_head_config_path or None,
        )
        if self.run_config.checkpoint.resume_from:
            validate_p_head_resume_config(resolved, checkpoint_dir=self.run_config.checkpoint.resume_from)
        return model

    def build_auxiliary_models(self, model: Any, tokenizer: Any) -> dict[str, Any]:
        from .q_model import build_q_model

        del model, tokenizer
        return {
            "q": build_q_model(
                self.recipe_config.q_config_path,
                init_device=self.run_config.model.init_device,
                torch_dtype=torch.float32,
                attn_implementation=self.run_config.model.attention,
            )
        }

    def after_parallelize(self, model: Any, auxiliary_models: Mapping[str, Any]) -> None:
        from .artifacts import load_p_head_weights

        load_p_head_weights(model, model_path=self.run_config.model.weights_path)
        self._q_model = auxiliary_models["q"]

    def save_recipe_assets(
        self,
        model: Any,
        tokenizer: Any,
        auxiliary_models: Mapping[str, Any],
        output_dir: str | Path,
    ) -> None:
        from .artifacts import save_recipe_assets

        del tokenizer
        save_recipe_assets(model, auxiliary_models["q"], output_dir)

    def export_auxiliary_weights(self, state_dicts: Mapping[str, Any], output_dir: str | Path) -> None:
        from .artifacts import save_q_weights

        save_q_weights(self._q_model, state_dicts["q"], Path(output_dir) / "q_model")

    def training_step(self, batch: TrainBatch, context: StepContext) -> StepMetrics:
        clean_input_ids = batch["input_ids"]
        if clean_input_ids.ndim != 2 or clean_input_ids.shape[1] != self.run_config.data.max_seq_len:
            raise ValueError("soldiff requires clean input_ids shaped [batch, data.max_seq_len]")
        if clean_input_ids.shape[0] == 0 or clean_input_ids.shape[1] == 0:
            raise ValueError("soldiff requires a nonempty clean token batch")
        if "loss_mask" in batch and not batch["loss_mask"].bool().all().item():
            raise ValueError("soldiff requires complete pretraining blocks without padding or prompt loss masks")

        batch_size, sequence_length = clean_input_ids.shape
        block_size = self.recipe_config.block_size
        device = clean_input_ids.device
        valid_token_mask = torch.ones_like(clean_input_ids, dtype=torch.bool)
        shared_mask_ratio = (1.0 - torch.rand((batch_size,), device=device, dtype=torch.float32)).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        clean_disallow_attention, document_ids = build_clean_blockwise_doc_attn_mask(
            clean_input_ids=clean_input_ids,
            block_size=block_size,
            eos_token_id=self.special_tokens["eos_token_id"],
        )
        clean_attention_mask = build_attention_mask_by_impl(
            disallow_attention_flag=clean_disallow_attention,
            attn_implementation=self.run_config.model.attention,
            enable_mixed_precision=True,
        )
        position_ids = torch.arange(sequence_length, device=device, dtype=torch.long).unsqueeze(0)
        q_position_ids = position_ids.expand(batch_size, -1).clone()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            q_output = context.forward_auxiliary(
                "q",
                {"input_ids": clean_input_ids, "attention_mask": clean_attention_mask, "position_ids": q_position_ids},
            )
            q_token_scores = _squeeze_token_scores_to_2d(
                _extract_token_scores(q_output), tuple(clean_input_ids.shape), "q_model"
            )
        del q_output, q_position_ids, clean_attention_mask, clean_disallow_attention
        groups = build_local_grouped_tensors(
            document_ids=document_ids, valid_token_mask=valid_token_mask, block_size=block_size
        )
        if not groups["has_active_groups"].item():
            loss = (
                module_zero_loss_anchor(context.auxiliary_model("q"), device)
                + module_zero_loss_anchor(context.module("p_head"), device)
            ) * 0.0
            scalar = loss.detach().new_zeros(())
            statistics = {
                name: scalar.new_zeros((block_size,)) if name.startswith("avg_rank_by_position_") else scalar
                for name in _RESULT_STATISTICS
            }
            losses = (loss, scalar, scalar, scalar)
        else:
            safe_indices = groups["dense_group_token_indices"].clamp_min(0)
            dense_qlogits = q_token_scores.reshape(-1)[safe_indices].masked_fill(
                ~groups["valid_group_slot_mask"], float("-inf")
            )
            del q_token_scores
            sampled = sample_two_local_orders_from_qlogits(
                dense_group_qlogits=dense_qlogits,
                dense_group_token_indices=groups["dense_group_token_indices"],
                group_token_counts=groups["group_token_counts"],
                group_batch_indices=groups["group_batch_indices"],
                shared_mask_ratio=shared_mask_ratio,
            )
            del shared_mask_ratio
            masked_inputs = build_two_masked_inputs_from_orders(
                clean_input_ids=clean_input_ids,
                sampled_group_token_orders=sampled["sampled_group_token_orders"],
                remaining_slot_mask=sampled["remaining_slot_mask"],
                mask_token_id=self.special_tokens["mask_token_id"],
            )
            combined_input_ids = torch.cat(
                [masked_inputs.reshape(2 * batch_size, sequence_length), clean_input_ids.repeat(2, 1)], dim=1
            )
            del masked_inputs
            combined_position_ids = torch.cat([position_ids, position_ids], dim=1).expand(2 * batch_size, -1).clone()
            full_positions = torch.arange(sequence_length * 2, device=device)
            block_disallow = ~block_diffusion_mask(
                None, None, full_positions[:, None], full_positions[None, :], block_size=block_size, n=sequence_length
            ).unsqueeze(0).unsqueeze(0)
            masked_disallow = build_block_diffusion_doc_disallow_mask_from_doc_ids(
                document_ids=document_ids.repeat(2, 1), block_diffusion_disallow_mask=block_disallow
            )
            del block_disallow, full_positions
            masked_attention_mask = build_attention_mask_by_impl(
                disallow_attention_flag=masked_disallow,
                attn_implementation=self.run_config.model.attention,
                enable_mixed_precision=True,
            )
            del masked_disallow
            outputs = context.forward(
                {
                    "input_ids": combined_input_ids,
                    "position_ids": combined_position_ids,
                    "attention_mask": masked_attention_mask,
                    "output_hidden_states": True,
                    "output_router_logits": False,
                }
            )
            noisy_logits = outputs.logits[:, :sequence_length]
            noisy_hidden_states = outputs.hidden_states[-1][:, :sequence_length]
            with context.forward_context():
                prior_logits = _squeeze_token_scores_to_2d(
                    context.module("p_head")(noisy_hidden_states), (2 * batch_size, sequence_length), "p_head"
                )
            safe_indices_by_path = safe_indices.unsqueeze(0).expand(2, -1, -1)
            dense_prior_logits = torch.gather(
                prior_logits.reshape(2, batch_size * sequence_length)
                .unsqueeze(1)
                .expand(-1, safe_indices.shape[0], -1),
                dim=2,
                index=safe_indices_by_path,
            ).masked_fill(~groups["valid_group_slot_mask"].unsqueeze(0).expand(2, -1, -1), float("-inf"))
            del outputs, noisy_hidden_states, prior_logits
            del masked_attention_mask, combined_input_ids, combined_position_ids
            objective = compute_local_residual_rloo_loss(
                dense_group_qlogits=sampled["dense_group_qlogits"],
                dense_group_prior_logits=dense_prior_logits,
                sampled_slot_orders=sampled["sampled_slot_orders"],
                sampled_group_token_orders=sampled["sampled_group_token_orders"],
                dense_group_token_indices=groups["dense_group_token_indices"],
                group_token_counts=sampled["group_token_counts"],
                visible_token_counts=sampled["visible_token_counts"],
                remaining_slot_mask=sampled["remaining_slot_mask"],
                group_batch_indices=sampled["group_batch_indices"],
                masked_noisy_logits=noisy_logits,
                clean_input_ids=clean_input_ids,
                block_size=block_size,
                suffix_reconstruction_tau=self.recipe_config.suffix_reconstruction_tau,
                lambda_align=self.recipe_config.lambda_align,
                lambda_recon=self.recipe_config.lambda_recon,
            )
            losses = objective[:4]
            statistics = dict(zip(_RESULT_STATISTICS, objective[4:], strict=True))

        # The shared context owns accumulation scaling; the method's divisor is independent.
        context.backward(losses[0], weight=1.0 / self.recipe_config.loss_backward_divisor)
        self._accumulate_statistics(context, losses, statistics)
        return StepMetrics(values={"loss": float(losses[0].detach().item()) / self.recipe_config.loss_backward_divisor})

    @staticmethod
    def _accumulate_statistics(context: StepContext, losses: Any, statistics: Mapping[str, Any]) -> None:
        saved = context.statistics.setdefault("soldiff", {})
        for name, value in zip(_LOSS_NAMES, losses, strict=True):
            saved[name] = saved.get(name, 0.0) + float(value.detach().item())
        saved["micro_batch_count"] = saved.get("micro_batch_count", 0.0) + 1.0
        for name, value in statistics.items():
            if name in _FLOAT_STATISTICS:
                # Accumulate these four scalar totals before float64 reduction.
                saved[name] = saved.get(name, 0.0) + float(value.detach().item())
                continue
            value = value.detach()
            if name not in saved:
                saved[name] = value.clone()
            elif name in _MAX_STATISTICS:
                saved[name] = torch.maximum(saved[name], value)
            else:
                saved[name] = saved[name] + value
        # The order-statistics denominator includes both sampled paths.
        saved["order_metric_count"] = saved.get("order_metric_count", 0.0) + float(statistics["group_count"].item())

    def finalize_step_metrics(self, values: Mapping[str, float], context: StepContext) -> dict[str, float]:
        saved = context.statistics.get("soldiff")
        if not saved:
            return dict(values)
        sums: dict[str, float] = {}
        maxima: dict[str, float] = {}
        for name, value in saved.items():
            if isinstance(value, torch.Tensor) and value.ndim:
                for index, item in enumerate(value):
                    sums[f"{name}/{index}"] = float(item.item())
            elif name in _MAX_STATISTICS:
                maxima[name] = float(value.item())
            else:
                sums[name] = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        sums["q_model_norm"] = _parameter_norm(context.auxiliary_model("q"))
        sums["p_head_norm"] = _parameter_norm(context.module("p_head"))
        sums["replica_count"] = 1.0
        # Reduce CE and count totals in float64; logits and order statistics
        # use float32.
        ce_totals = {
            name: sums.pop(name) for name in ("ce_sum", "weighted_ce_sum", "active_token_count", "group_count")
        }
        ce_totals = context.reduce_statistics(ce_totals, reduction="sum", dtype="float64")
        sums = context.reduce_statistics(sums, reduction="sum")
        sums.update(ce_totals)
        maxima = context.reduce_statistics(maxima, reduction="max")
        metrics = dict(values)
        micro_batch_count = max(sums["micro_batch_count"], 1.0)
        divisor = self.recipe_config.loss_backward_divisor
        for name in _LOSS_NAMES:
            metrics[name] = sums[name] / micro_batch_count / divisor
        metrics["ce_per_token"] = sums["ce_sum"] / max(sums["active_token_count"], 1.0)
        metrics["q_weighted_ce_per_token"] = sums["weighted_ce_sum"] / max(sums["group_count"], 1.0)
        for prefix, output_prefix in (("q", "q"), ("prior", "p")):
            count = max(sums[f"{prefix}_logits_count"], 1.0)
            mean = sums[f"{prefix}_logits_sum"] / count
            variance = max(sums[f"{prefix}_logits_sq_sum"] / count - mean * mean, 0.0)
            metrics[f"{output_prefix}_logits_std"] = variance**0.5
            metrics[f"{output_prefix}_logits_abs_max"] = maxima[f"{prefix}_logits_abs_max"]
        metrics["order_stats/inversion_norm"] = sums["inversion_norm_sum"] / max(sums["order_metric_count"], 1.0)
        for index in range(self.recipe_config.block_size):
            metrics[f"order_stats/avg_rank_by_position/{index}"] = sums[f"avg_rank_by_position_sum/{index}"] / max(
                sums[f"avg_rank_by_position_count/{index}"], 1.0
            )
        for name in ("q_model_norm", "p_head_norm"):
            metrics[name] = sums[name] / sums["replica_count"]
        return metrics


__all__ = ["SolDiffTrainer"]
