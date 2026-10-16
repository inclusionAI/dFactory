from dfactory.config import RunConfig
from dfactory.trainers import BaseTrainer, StepMetrics, TrainerState


class FakeTrainer(BaseTrainer):
    def training_step(self, batch, context):
        assert context.accumulation_steps == 2
        return StepMetrics({"loss": float(batch)}, weight=1.0)


class FakeOptimizer:
    def __init__(self) -> None:
        self.steps = 0
        self.zeroes = 0

    def zero_grad(self) -> None:
        self.zeroes += 1

    def step(self) -> None:
        self.steps += 1


class FakeScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def get_last_lr(self):
        return [0.01]


def test_trainer_loop_runs_algorithm_step_and_service_helpers() -> None:
    optimizer = FakeOptimizer()
    scheduler = FakeScheduler()
    state = TrainerState()
    seen = []
    trainer = FakeTrainer(
        run_config=RunConfig(),
        recipe_config=None,
        data_factory=lambda **kwargs: None,
    )

    trainer._run_training_loop(
        batches=[[1, 3], [2, 4]],
        model=object(),
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        max_steps=2,
        device="cuda",
        metric_sink=lambda step, values: seen.append((step, values)),
    )

    assert optimizer.steps == scheduler.steps == state.global_step == 2
    assert optimizer.zeroes == 2
    assert [item[1]["loss"] for item in seen] == [2.0, 3.0]
