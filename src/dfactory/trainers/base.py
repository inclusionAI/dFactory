"""Shared lifecycle for dFactory trainer implementations."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from dfactory.config import RunConfig
from dfactory.data import TrainBatch, get_dataset
from dfactory.trainers.context import StepContext
from dfactory.trainers.metrics import MetricAccumulator, StepMetrics
from dfactory.trainers.state import TrainerState


class BaseTrainer(ABC):
    """Own the common distributed training lifecycle for one recipe."""

    def __init__(
        self,
        *,
        run_config: RunConfig,
        recipe_config: Any,
        data_factory: Callable[..., Any],
    ) -> None:
        self.run_config = run_config
        self.recipe_config = recipe_config
        self.data_factory = data_factory
        self.special_tokens: Mapping[str, int] = {}

    @abstractmethod
    def training_step(self, batch: TrainBatch, context: StepContext) -> StepMetrics:
        """Run the algorithm-specific forward and backward work for one micro-batch."""

    def prepare_main_model(self, model: Any, tokenizer: Any) -> Any:
        """Attach recipe-owned trainable modules before FSDP and optimizer setup."""
        return model

    def build_auxiliary_models(self, model: Any, tokenizer: Any) -> Mapping[str, Any]:
        """Build auxiliary models on the configured initialization device."""
        return {}

    def after_parallelize(self, model: Any, auxiliary_models: Mapping[str, Any]) -> None:
        """Optionally load sidecar weights after model materialization."""

    def data_loader_generator(self) -> Any:
        """Optionally isolate iterator/worker seeds from model randomness."""
        return None

    def save_recipe_assets(
        self, model: Any, tokenizer: Any, auxiliary_models: Mapping[str, Any], output_dir: Path
    ) -> None:
        """Save resolved recipe structure independently of external input paths."""

    def export_auxiliary_weights(self, state_dicts: Mapping[str, Any], output_dir: Path) -> None:
        """Optionally export auxiliary CPU state dictionaries on rank zero."""

    def finalize_step_metrics(self, values: dict[str, float], context: StepContext) -> dict[str, float]:
        """Convert recipe sufficient statistics into optimizer-step metrics."""
        return values

    def train(self) -> None:
        """Build runtime resources and execute the configured training run."""
        from torch.utils.data import DataLoader, DistributedSampler

        from dfactory.models import get_model
        from dfactory.runtime.veomni import VeOmniGateway

        config = self.run_config
        gateway = VeOmniGateway()
        runtime = gateway.initialize(config.distributed, seed=config.run.seed)
        denominator = config.optimization.micro_batch_size * runtime.data_parallel_size
        if config.optimization.global_batch_size % denominator:
            raise ValueError(
                "optimization.global_batch_size must be divisible by micro_batch_size * data_parallel_size"
            )
        accumulation_steps = config.optimization.global_batch_size // denominator

        model_spec = get_model(config.model.provider)
        model_provider = model_spec.provider_type()
        tokenizer = model_provider.load_tokenizer(config.model)
        model = model_provider.build_model(config.model)
        model = self.prepare_main_model(model, tokenizer)
        self.special_tokens = model_provider.special_tokens(tokenizer, model.config, config.model)
        auxiliary_models = dict(self.build_auxiliary_models(model, tokenizer))
        for name in auxiliary_models:
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"invalid auxiliary model name: {name!r}")

        source_spec = get_dataset(config.data.source)
        source = source_spec.factory(config.data.path)
        recipe_data = self.data_factory(
            source=source,
            tokenizer=tokenizer,
            data_config=config.data,
            recipe_config=self.recipe_config,
            special_tokens=self.special_tokens,
        )
        sampler = DistributedSampler(
            recipe_data.dataset,
            num_replicas=runtime.data_parallel_size,
            rank=runtime.data_parallel_rank,
            shuffle=True,
            seed=config.run.seed,
        )
        dataloader = DataLoader(
            recipe_data.dataset,
            batch_size=config.optimization.micro_batch_size,
            sampler=sampler,
            num_workers=config.data.num_workers,
            drop_last=True,
            collate_fn=recipe_data.collate_fn,
            generator=self.data_loader_generator(),
        )

        model = gateway.parallelize(
            model,
            config.model,
            config.distributed,
            sharding_plan=model_provider.parallel_policy(model),
        )
        model.train()
        for name, auxiliary in auxiliary_models.items():
            auxiliary_models[name] = gateway.parallelize(
                auxiliary, config.model, config.distributed, weights_path=None, strict=False
            )
            auxiliary_models[name].train()
        self.after_parallelize(model, auxiliary_models)
        optimizer, scheduler = gateway.build_optimization(
            model,
            config.optimization,
            train_steps=config.run.max_steps,
        )
        auxiliary_optimizers = {}
        for name, auxiliary in auxiliary_models.items():
            auxiliary_optimizers[name], _ = gateway.build_optimization(
                auxiliary, config.optimization, train_steps=config.run.max_steps
            )
        forward_context, backward_context = gateway.activation_contexts(config.distributed)

        output_dir = Path(config.run.output_dir)
        if runtime.global_rank == 0:
            self._save_resolved_config(config, output_dir)
            model_provider.save_assets(model, tokenizer, output_dir / "model_assets")
            self.save_recipe_assets(model, tokenizer, auxiliary_models, output_dir / "model_assets")

        checkpoint_io = gateway.checkpoint_io(config.distributed.mode)
        state = TrainerState()
        if config.checkpoint.resume_from:
            state = self._restore_checkpoint(
                checkpoint_io,
                config.checkpoint.resume_from,
                model,
                optimizer,
                scheduler,
            )
            self._restore_auxiliary_checkpoints(
                checkpoint_io, Path(config.checkpoint.resume_from), auxiliary_models, auxiliary_optimizers, state
            )

        checkpoint_root = output_dir / "checkpoints"
        last_saved_step: int | None = None

        def save_checkpoint(current_state: TrainerState) -> None:
            nonlocal last_saved_step
            checkpoint_io.save(
                checkpoint_root,
                self._checkpoint_state(model, optimizer, scheduler, current_state, config),
                global_step=current_state.global_step,
            )
            checkpoint_dir = checkpoint_root / f"global_step_{current_state.global_step}"
            self._save_auxiliary_checkpoints(
                checkpoint_io, checkpoint_dir, auxiliary_models, auxiliary_optimizers, current_state
            )
            if runtime.global_rank == 0:
                self._save_resolved_config(config, checkpoint_dir)
                model_provider.save_assets(model, tokenizer, checkpoint_dir / "model_assets")
                self.save_recipe_assets(model, tokenizer, auxiliary_models, checkpoint_dir / "model_assets")
            last_saved_step = current_state.global_step

        metric_sink = (
            self._stdout_metric_sink if runtime.global_rank == 0 and "stdout" in config.logging.report_to else None
        )
        clip_gradients = partial(gateway.clip_gradients, max_norm=config.optimization.max_grad_norm)
        steps_per_epoch = len(dataloader) // accumulation_steps
        if steps_per_epoch == 0:
            raise ValueError("the dataset does not contain one complete global batch")
        if state.step_in_epoch > steps_per_epoch:
            raise ValueError("checkpoint step_in_epoch exceeds the available optimizer steps")
        if state.step_in_epoch == steps_per_epoch:
            state.next_epoch()

        for epoch in range(state.epoch, config.run.num_train_epochs):
            state.epoch = epoch
            sampler.set_epoch(epoch)
            batch_groups = self._batch_groups(dataloader, accumulation_steps, skip=state.step_in_epoch)
            before_epoch = state.global_step
            self._run_training_loop(
                batches=batch_groups,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                max_steps=config.run.max_steps,
                device=runtime.device,
                move_batch=gateway.move_batch,
                clip_gradients=clip_gradients,
                reduce_loss=gateway.reduce_loss,
                forward_context_factory=lambda: forward_context,
                backward_context_factory=lambda: backward_context,
                metric_sink=metric_sink,
                checkpoint_callback=save_checkpoint,
                save_every_steps=config.checkpoint.save_every_steps,
                log_every_steps=config.logging.log_every_steps,
                auxiliary_models=auxiliary_models,
                auxiliary_optimizers=auxiliary_optimizers,
                reduce_statistics=getattr(gateway, "reduce_statistics", None),
            )
            if state.global_step >= config.run.max_steps:
                break
            if state.global_step == before_epoch:
                break
            state.next_epoch()
            if config.checkpoint.save_every_epochs and state.epoch % config.checkpoint.save_every_epochs == 0:
                save_checkpoint(state)

        if state.global_step > 0 and last_saved_step != state.global_step:
            save_checkpoint(state)
        gateway.synchronize()
        if runtime.global_rank == 0 and state.global_step > 0 and config.checkpoint.save_hf_weights:
            checkpoint_dir = checkpoint_root / f"global_step_{state.global_step}"
            model_provider.save_weights(
                model,
                tokenizer,
                checkpoint_io.model_state_dict(checkpoint_dir),
                checkpoint_dir / "hf_ckpt",
            )
            if auxiliary_models:
                auxiliary_states = {
                    name: checkpoint_io.model_state_dict(self._auxiliary_checkpoint_path(checkpoint_dir, name, state))
                    for name in auxiliary_models
                }
                self.export_auxiliary_weights(auxiliary_states, checkpoint_dir / "hf_ckpt")

    def _run_training_loop(
        self,
        *,
        batches: Iterable[Sequence[Any]],
        model: Any,
        optimizer: Any,
        scheduler: Any,
        state: TrainerState,
        max_steps: int,
        device: str,
        move_batch: Callable[[Any, str], Any] = lambda batch, _device: batch,
        clip_gradients: Callable[[Any], float] = lambda _model: 0.0,
        reduce_loss: Callable[[Any, Any], Any] | None = None,
        forward_context_factory: Callable[[], Any] | None = None,
        backward_context_factory: Callable[[], Any] | None = None,
        metric_sink: Callable[[int, dict[str, float]], None] | None = None,
        checkpoint_callback: Callable[[TrainerState], None] | None = None,
        save_every_steps: int | None = None,
        log_every_steps: int = 1,
        auxiliary_models: Mapping[str, Any] | None = None,
        auxiliary_optimizers: Mapping[str, Any] | None = None,
        reduce_statistics: Callable[..., dict[str, float]] | None = None,
    ) -> TrainerState:
        context_kwargs: dict[str, Any] = {}
        if forward_context_factory is not None:
            context_kwargs["forward_context_factory"] = forward_context_factory
        if backward_context_factory is not None:
            context_kwargs["backward_context_factory"] = backward_context_factory
        auxiliary_models = dict(auxiliary_models or {})
        auxiliary_optimizers = dict(auxiliary_optimizers or {})
        if auxiliary_models.keys() != auxiliary_optimizers.keys():
            raise ValueError("every auxiliary model must have exactly one optimizer")

        for micro_batches in batches:
            if state.global_step >= max_steps:
                break
            if not micro_batches:
                continue

            optimizer.zero_grad()
            for auxiliary_optimizer in auxiliary_optimizers.values():
                auxiliary_optimizer.zero_grad()
            context = StepContext(
                model=model,
                state=state,
                accumulation_steps=len(micro_batches),
                device=device,
                reduce_loss=reduce_loss,
                auxiliary_models=auxiliary_models,
                reduce_statistics=reduce_statistics,
                **context_kwargs,
            )
            metrics = MetricAccumulator()
            for micro_batch in micro_batches:
                metrics.add(self.training_step(move_batch(micro_batch, device), context))

            # Synchronize auxiliary learning rates, then clip and update each
            # auxiliary model before updating the main model and its scheduler.
            for name, auxiliary_optimizer in auxiliary_optimizers.items():
                source_groups = optimizer.param_groups
                target_groups = auxiliary_optimizer.param_groups
                if source_groups:
                    for index, target in enumerate(target_groups):
                        target["lr"] = source_groups[min(index, len(source_groups) - 1)]["lr"]
                clip_gradients(auxiliary_models[name])
                auxiliary_optimizer.step()
                auxiliary_optimizer.zero_grad()
            grad_norm = clip_gradients(model)
            optimizer.step()
            scheduler.step()
            state.advance()

            values = metrics.compute()
            values["grad_norm"] = float(grad_norm)
            values["lr"] = max(float(value) for value in scheduler.get_last_lr())
            values = self.finalize_step_metrics(values, context)
            if metric_sink is not None and state.global_step % log_every_steps == 0:
                metric_sink(state.global_step, values)
            if save_every_steps and checkpoint_callback and state.global_step % save_every_steps == 0:
                checkpoint_callback(state)

        return state

    @staticmethod
    def _auxiliary_checkpoint_path(checkpoint_dir: Path, name: str, state: TrainerState) -> Path:
        return checkpoint_dir / "auxiliary" / name / f"global_step_{state.global_step}"

    @classmethod
    def _save_auxiliary_checkpoints(
        cls,
        checkpoint_io: Any,
        checkpoint_dir: Path,
        models: Mapping[str, Any],
        optimizers: Mapping[str, Any],
        state: TrainerState,
    ) -> None:
        for name, auxiliary in models.items():
            checkpoint_io.save(
                checkpoint_dir / "auxiliary" / name,
                {
                    "model": auxiliary,
                    "optimizer": optimizers[name],
                    "extra_state": {"global_step": state.global_step},
                },
                global_step=state.global_step,
            )

    @classmethod
    def _restore_auxiliary_checkpoints(
        cls,
        checkpoint_io: Any,
        checkpoint_dir: Path,
        models: Mapping[str, Any],
        optimizers: Mapping[str, Any],
        state: TrainerState,
    ) -> None:
        for name, auxiliary in models.items():
            path = cls._auxiliary_checkpoint_path(checkpoint_dir, name, state)
            loaded = checkpoint_io.load(path, {"model": auxiliary, "optimizer": optimizers[name], "extra_state": {}})
            if loaded["extra_state"]["global_step"] != state.global_step:
                raise ValueError(f"auxiliary model {name!r} checkpoint step does not match the main model")

    @staticmethod
    def _batch_groups(batches: Iterable[Any], size: int, *, skip: int = 0) -> Iterator[tuple[Any, ...]]:
        if size <= 0:
            raise ValueError("gradient accumulation steps must be positive")
        group: list[Any] = []
        for index, batch in enumerate(batches):
            if index < skip * size:
                continue
            group.append(batch)
            if len(group) == size:
                yield tuple(group)
                group.clear()

    @staticmethod
    def _stdout_metric_sink(step: int, values: dict[str, float]) -> None:
        print(json.dumps({"step": step, **values}, sort_keys=True), flush=True)

    @staticmethod
    def _checkpoint_state(
        model: Any,
        optimizer: Any,
        scheduler: Any,
        state: TrainerState,
        config: RunConfig,
    ) -> dict[str, Any]:
        import torch

        return {
            "model": model,
            "optimizer": optimizer,
            "extra_state": {
                "training_state": {"global_step": state.global_step},
                "dataloader_state": {
                    "epoch": state.epoch,
                    "optimizer_step_in_epoch": state.step_in_epoch,
                },
                "scheduler": scheduler.state_dict(),
                "random_state": {
                    "cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                },
                "resolved_config": config.to_dict(),
            },
        }

    @staticmethod
    def _restore_checkpoint(
        checkpoint_io: Any,
        path: str,
        model: Any,
        optimizer: Any,
        scheduler: Any,
    ) -> TrainerState:
        import torch

        loaded = checkpoint_io.load(path, {"model": model, "optimizer": optimizer, "extra_state": {}})
        extra_state = loaded["extra_state"]
        scheduler.load_state_dict(extra_state["scheduler"])
        random_state = extra_state["random_state"]
        torch.set_rng_state(random_state["cpu"])
        torch.cuda.set_rng_state_all(random_state["cuda"])
        training_state = extra_state["training_state"]
        dataloader_state = extra_state["dataloader_state"]
        return TrainerState(
            epoch=dataloader_state["epoch"],
            step_in_epoch=dataloader_state["optimizer_step_in_epoch"],
            global_step=training_state["global_step"],
        )

    @staticmethod
    def _save_resolved_config(config: RunConfig, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved.yaml").write_text(config.to_yaml(), encoding="utf-8")


__all__ = ["BaseTrainer"]
