from types import SimpleNamespace

import pytest
import torch

from dfactory.config import RunConfig
from dfactory.trainers import BaseTrainer, StepContext, StepMetrics, TrainerState


class ScalarModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, value, **kwargs):
        return self.weight * value


class Scorer(ScalarModel):
    # Unlike the main HF model this intentionally does not accept HF kwargs.
    def forward(self, value):
        return self.weight * value


class TwoModelTrainer(BaseTrainer):
    def training_step(self, batch, context):
        main = context.forward({"value": batch})
        q = context.forward_auxiliary("q", {"value": batch})
        loss = main + q
        context.backward(loss, weight=1.0 / 4096.0)
        context.statistics["micro_batches"] = context.statistics.get("micro_batches", 0) + 1
        return StepMetrics({"loss": float(loss.detach())})

    def finalize_step_metrics(self, values, context):
        assert context.global_step == 1
        assert context.statistics["micro_batches"] == 2
        return values


def test_auxiliary_step_preserves_accumulation_divisor_lr_and_update_order():
    main, q = ScalarModel(2), Scorer(3)
    events = []

    class Optimizer(torch.optim.SGD):
        def __init__(self, model, name, lr):
            super().__init__(model.parameters(), lr=lr)
            self.name = name

        def step(self, closure=None):
            events.append(f"step:{self.name}")
            return super().step(closure)

    main_optimizer = Optimizer(main, "main", 0.125)
    q_optimizer = Optimizer(q, "q", 0.75)
    scheduler = SimpleNamespace(
        step=lambda: events.append("scheduler"),
        get_last_lr=lambda: [main_optimizer.param_groups[0]["lr"]],
    )
    observed_gradients = {}

    def clip(model):
        name = "main" if model is main else "q"
        events.append(f"clip:{name}")
        observed_gradients[name] = model.weight.grad.item()
        return model.weight.grad.abs().item()

    trainer = TwoModelTrainer(run_config=RunConfig(), recipe_config=None, data_factory=lambda **kwargs: None)
    trainer._run_training_loop(
        batches=[(torch.tensor(2.0), torch.tensor(6.0))],
        model=main,
        optimizer=main_optimizer,
        scheduler=scheduler,
        state=TrainerState(),
        max_steps=1,
        device="cpu",
        clip_gradients=clip,
        auxiliary_models={"q": q},
        auxiliary_optimizers={"q": q_optimizer},
    )
    assert events == ["clip:q", "step:q", "clip:main", "step:main", "scheduler"]
    assert observed_gradients == {"main": 4.0 / 4096.0, "q": 4.0 / 4096.0}
    assert q_optimizer.param_groups[0]["lr"] == 0.125
    torch.testing.assert_close(main.weight, torch.tensor(2.0 - 0.125 * 4.0 / 4096.0), rtol=0, atol=0)
    torch.testing.assert_close(q.weight, torch.tensor(3.0 - 0.125 * 4.0 / 4096.0), rtol=0, atol=0)


def test_context_auxiliary_signature_and_registered_head_access():
    main = ScalarModel(2)
    main.p_head = Scorer(4)
    q = Scorer(3)
    calls = []

    def reduce(values, *, reduction):
        calls.append(reduction)
        return {key: value * 2 for key, value in values.items()}

    context = StepContext(
        model=main,
        auxiliary_models={"q": q},
        state=TrainerState(),
        accumulation_steps=1,
        device="cpu",
        reduce_statistics=reduce,
    )
    assert context.module() is main
    assert context.module("p_head") is main.p_head
    assert context.auxiliary_model("q") is q
    assert context.forward_auxiliary("q", {"value": torch.tensor(2.0)}).item() == 6
    assert context.reduce_statistics({"tokens": 3}, reduction="sum") == {"tokens": 6}
    assert calls == ["sum"]
    with pytest.raises(KeyError, match="unknown auxiliary"):
        context.auxiliary_model("missing")
    with pytest.raises(ValueError, match="reduction"):
        context.reduce_statistics({}, reduction="mean")


def test_auxiliary_checkpoint_pairs_models_optimizers_and_steps(tmp_path):
    main_checkpoint = tmp_path / "global_step_12"
    q, optimizer = object(), object()
    saved = []

    class Checkpoints:
        def save(self, path, payload, *, global_step):
            saved.append((path, payload, global_step))

        def load(self, path, payload):
            assert path == main_checkpoint / "auxiliary" / "q" / "global_step_12"
            assert payload["model"] is q
            assert payload["optimizer"] is optimizer
            return {"extra_state": {"global_step": self.restored_step}}

    io = Checkpoints()
    state = TrainerState(global_step=12)
    BaseTrainer._save_auxiliary_checkpoints(io, main_checkpoint, {"q": q}, {"q": optimizer}, state)
    path, payload, step = saved[0]
    assert path == main_checkpoint / "auxiliary" / "q"
    assert payload == {"model": q, "optimizer": optimizer, "extra_state": {"global_step": 12}}
    assert step == 12
    io.restored_step = 12
    BaseTrainer._restore_auxiliary_checkpoints(io, main_checkpoint, {"q": q}, {"q": optimizer}, state)
    io.restored_step = 11
    with pytest.raises(ValueError, match="does not match"):
        BaseTrainer._restore_auxiliary_checkpoints(io, main_checkpoint, {"q": q}, {"q": optimizer}, state)


def test_auxiliary_model_without_optimizer_is_rejected():
    trainer = TwoModelTrainer(run_config=RunConfig(), recipe_config=None, data_factory=lambda **kwargs: None)
    with pytest.raises(ValueError, match="exactly one optimizer"):
        trainer._run_training_loop(
            batches=[],
            model=ScalarModel(1),
            optimizer=object(),
            scheduler=object(),
            state=TrainerState(),
            max_steps=1,
            device="cpu",
            auxiliary_models={"q": Scorer(2)},
        )


def test_actual_cpu_dcp_restores_main_p_q_optimizers_and_next_update(monkeypatch, tmp_path):
    # CPU checkpoint round trip with device synchronization disabled and a
    # single-process rank supplied for DCP.
    import veomni.checkpoint.dcp_checkpointer as checkpoint_module

    from dfactory.runtime.veomni.checkpoint import CheckpointIO

    monkeypatch.setattr(checkpoint_module, "empty_cache", lambda: None)
    monkeypatch.setattr(checkpoint_module, "synchronize", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    main, q = ScalarModel(2), Scorer(3)
    main.p_head = Scorer(4)
    optimizer = torch.optim.AdamW(main.parameters(), lr=0.01)
    q_optimizer = torch.optim.AdamW(q.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    def update():
        optimizer.zero_grad()
        q_optimizer.zero_grad()
        (main.weight.square() + main.p_head.weight.square() + q.weight.square()).backward()
        q_optimizer.step()
        optimizer.step()
        scheduler.step()

    update()
    state = TrainerState(global_step=1, step_in_epoch=1)
    io = CheckpointIO("fsdp2")
    main_path = tmp_path / "global_step_1"
    io.save(tmp_path, BaseTrainer._checkpoint_state(main, optimizer, scheduler, state, RunConfig()), global_step=1)
    BaseTrainer._save_auxiliary_checkpoints(io, main_path, {"q": q}, {"q": q_optimizer}, state)
    expected_rng = torch.rand(3)
    update()
    expected_main = {name: tensor.clone() for name, tensor in main.state_dict().items()}
    expected_q = {name: tensor.clone() for name, tensor in q.state_dict().items()}
    with torch.no_grad():
        for parameter in [*main.parameters(), *q.parameters()]:
            parameter.add_(100)
    optimizer.param_groups[0]["lr"] = 99
    q_optimizer.param_groups[0]["lr"] = 99
    restored = BaseTrainer._restore_checkpoint(io, main_path, main, optimizer, scheduler)
    BaseTrainer._restore_auxiliary_checkpoints(io, main_path, {"q": q}, {"q": q_optimizer}, restored)
    assert restored.global_step == restored.step_in_epoch == 1
    torch.testing.assert_close(torch.rand(3), expected_rng, rtol=0, atol=0)
    assert optimizer.param_groups[0]["lr"] == q_optimizer.param_groups[0]["lr"] == 0.01
    update()
    for name, tensor in main.state_dict().items():
        torch.testing.assert_close(tensor, expected_main[name], rtol=0, atol=0)
    for name, tensor in q.state_dict().items():
        torch.testing.assert_close(tensor, expected_q[name], rtol=0, atol=0)
