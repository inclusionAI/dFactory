from contextlib import nullcontext
from types import SimpleNamespace

import torch

import dfactory.models
import dfactory.runtime.veomni
import dfactory.trainers.base as trainer_module
from dfactory.config import DataConfig, ModelConfig, OptimizationConfig, RunConfig, RunSettings
from dfactory.recipes import RecipeData
from dfactory.runtime import RuntimeContext
from dfactory.trainers import BaseTrainer, StepMetrics, TrainerState


def test_batch_groups_skip_resumed_optimizer_steps_and_drop_incomplete_tail() -> None:
    groups = list(BaseTrainer._batch_groups(range(9), 3, skip=1))
    assert groups == [(3, 4, 5), (6, 7, 8)]


class FakeTrainer(BaseTrainer):
    def training_step(self, batch, context):
        loss = context.forward({"value": batch["value"]}).sum()
        context.backward(loss)
        return StepMetrics({"loss": float(loss.detach())})


class FakeModel(torch.nn.Module):
    config = object()

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, value, **kwargs):
        del kwargs
        return value * self.weight


class FakeProvider:
    def __init__(self) -> None:
        self.exported_to = None
        self.asset_directories = []

    def load_tokenizer(self, config):
        return object()

    def build_model(self, config):
        return FakeModel()

    def special_tokens(self, tokenizer, model_config, config):
        return {"mask_token_id": 9, "eos_token_id": 8, "pad_token_id": 0}

    def parallel_policy(self, model):
        return None

    def save_assets(self, model, tokenizer, output_dir):
        self.asset_directories.append(output_dir)

    def save_weights(self, model, tokenizer, state_dict, output_dir):
        self.exported_to = output_dir


class FakeScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def get_last_lr(self):
        return [0.1]

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = state["steps"]


class FakeCheckpointIO:
    def __init__(self) -> None:
        self.saved_steps = []

    def save(self, path, state, *, global_step):
        self.saved_steps.append(global_step)

    def model_state_dict(self, path):
        return {}


class FakeGateway:
    def __init__(self) -> None:
        self.checkpoints = FakeCheckpointIO()
        self.synchronized = False

    def initialize(self, config, *, seed):
        return RuntimeContext("cuda", 0, 0, 2, 2, 0, 1, 1, 1)

    def parallelize(self, model, model_config, distributed_config, *, sharding_plan=None):
        self.model = model
        return model

    def build_optimization(self, model, config, *, train_steps):
        return torch.optim.SGD(model.parameters(), lr=0.1), FakeScheduler()

    def activation_contexts(self, config):
        return nullcontext(), nullcontext()

    def checkpoint_io(self, distributed_mode):
        return self.checkpoints

    def move_batch(self, batch, device):
        return batch

    def reduce_loss(self, loss, token_count):
        return loss

    def clip_gradients(self, model, max_norm):
        return 0.0

    def synchronize(self):
        self.synchronized = True


def test_base_trainer_assembles_a_complete_training_run(monkeypatch, tmp_path) -> None:
    gateway = FakeGateway()
    provider = FakeProvider()
    monkeypatch.setattr(
        dfactory.models,
        "get_model",
        lambda name: SimpleNamespace(provider_type=lambda: provider),
    )
    monkeypatch.setattr(
        trainer_module,
        "get_dataset",
        lambda name: SimpleNamespace(factory=lambda path: [{"value": index} for index in range(4)]),
    )
    monkeypatch.setattr(dfactory.runtime.veomni, "VeOmniGateway", lambda: gateway)
    config = RunConfig(
        run=RunSettings(output_dir=str(tmp_path), max_steps=2, num_train_epochs=1),
        model=ModelConfig(config_path="config", weights_path="weights", tokenizer_path="tokenizer"),
        data=DataConfig(path="data", max_seq_len=4),
        optimization=OptimizationConfig(global_batch_size=2, micro_batch_size=1),
    )
    trainer = FakeTrainer(
        run_config=config,
        recipe_config=None,
        data_factory=lambda *, source, **kwargs: RecipeData(
            [{"value": torch.tensor(float(record["value"]))} for record in source]
        ),
    )

    trainer.train()

    assert gateway.checkpoints.saved_steps == [2]
    assert gateway.synchronized
    assert gateway.model.weight.item() != 1.0
    assert provider.exported_to == tmp_path / "checkpoints" / "global_step_2" / "hf_ckpt"
    assert provider.asset_directories == [
        tmp_path / "model_assets",
        tmp_path / "checkpoints" / "global_step_2" / "model_assets",
    ]
    assert (tmp_path / "resolved.yaml").is_file()
    assert (tmp_path / "checkpoints" / "global_step_2" / "resolved.yaml").is_file()


def test_checkpoint_state_records_cpu_and_cuda_rng(monkeypatch) -> None:
    expected_cuda_rng_state = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: expected_cuda_rng_state)
    config = RunConfig(
        model=ModelConfig(config_path="config", weights_path="weights", tokenizer_path="tokenizer"),
        data=DataConfig(path="data"),
    )

    checkpoint = BaseTrainer._checkpoint_state(
        object(),
        object(),
        FakeScheduler(),
        TrainerState(global_step=2),
        config,
    )

    assert torch.equal(checkpoint["extra_state"]["random_state"]["cpu"], torch.get_rng_state())
    assert checkpoint["extra_state"]["random_state"]["cuda"] is expected_cuda_rng_state
    assert checkpoint["extra_state"]["resolved_config"] == config.to_dict()
    assert checkpoint["extra_state"]["dataloader_state"] == {"epoch": 0, "optimizer_step_in_epoch": 0}


def test_restore_checkpoint_restores_scheduler_rng_and_training_state(monkeypatch) -> None:
    scheduler = FakeScheduler()
    expected_rng_state = torch.get_rng_state()
    expected_cuda_rng_state = [torch.tensor([4, 5, 6], dtype=torch.uint8)]
    restored_cuda_rng_state = []
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored_cuda_rng_state.append)

    class LoadedCheckpoint:
        def load(self, path, state):
            assert path == "checkpoint"
            assert set(state) == {"model", "optimizer", "extra_state"}
            return {
                "extra_state": {
                    "scheduler": {"steps": 3},
                    "training_state": {"global_step": 9},
                    "dataloader_state": {"epoch": 2, "optimizer_step_in_epoch": 4},
                    "random_state": {"cpu": expected_rng_state, "cuda": expected_cuda_rng_state},
                    "resolved_config": {},
                }
            }

    state = BaseTrainer._restore_checkpoint(LoadedCheckpoint(), "checkpoint", object(), object(), scheduler)

    assert scheduler.steps == 3
    assert (state.epoch, state.step_in_epoch, state.global_step) == (2, 4, 9)
    assert torch.equal(torch.get_rng_state(), expected_rng_state)
    assert len(restored_cuda_rng_state) == 1
    assert restored_cuda_rng_state[0] is expected_cuda_rng_state
