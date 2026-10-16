"""Behavior tests for the built-in ESTF trainer."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dfactory.config import DataConfig, RunConfig, load_config
from dfactory.recipes.estf import ESTFConfig, ESTFTrainer, build_data, recipe_spec


class RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)
        self.inputs: list[torch.Tensor] = []
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.bias[0] = 1.0

    def forward(self, input_ids, **kwargs):
        del kwargs
        self.inputs.append(input_ids.detach().clone())
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


class RecordingStepContext:
    global_step = 0
    epoch = 0
    accumulation_steps = 2
    device = "cpu"

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.backward_calls: list[tuple[torch.Tensor, float]] = []

    def forward(self, model_inputs):
        return self.model(**dict(model_inputs))

    def backward(self, loss, *, weight: float = 1.0) -> None:
        self.backward_calls.append((loss.detach().clone(), weight))
        (loss * weight / self.accumulation_steps).backward()

    def reduce_loss(self, loss, token_count):
        del token_count
        return loss


def _batch(example: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.unsqueeze(0) for name, value in example.items()}


def _trainer(config: ESTFConfig, *, special_tokens: dict[str, int]) -> ESTFTrainer:
    trainer = ESTFTrainer(
        run_config=RunConfig(),
        recipe_config=config,
        data_factory=build_data,
    )
    trainer.special_tokens = special_tokens
    return trainer


def test_estf_config_is_strict_and_validates_owned_parameters() -> None:
    assert ESTFConfig.from_mapping({}).max_stages == 4
    assert ESTFConfig.from_mapping({"confidence_source": "maximum"}).confidence_source == "maximum"
    assert ESTFConfig.from_mapping({"initial_known_ratio": [0.1, 0.6]}).initial_known_ratio == (0.1, 0.6)
    with pytest.raises(ValueError, match="unknown ESTF parameter"):
        ESTFConfig.from_mapping({"rpt_deterministic_schedule": True})
    with pytest.raises(ValueError, match="max_stages must be a positive integer"):
        ESTFConfig.from_mapping({"max_stages": 0})
    with pytest.raises(ValueError, match="confidence_source"):
        ESTFConfig.from_mapping({"confidence_source": "entropy"})
    with pytest.raises(ValueError, match="initial_known_ratio"):
        ESTFConfig.from_mapping({"initial_known_ratio": [0.6, 0.1]})
    with pytest.raises(ValueError, match="initial_known_schedule_power"):
        ESTFConfig.from_mapping({"initial_known_schedule_power": 0.0})


def test_estf_data_factory_uses_token_records_and_preserves_boundaries() -> None:
    recipe_data = build_data(
        source=[{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}],
        tokenizer=None,
        data_config=DataConfig(
            path="unused",
            format="token_records",
            text_key="input_ids",
            max_seq_len=8,
            packing=True,
        ),
        recipe_config=ESTFConfig(block_size=4),
        special_tokens={"pad_token_id": 0},
    )

    example = recipe_data.dataset[0]
    assert example["input_ids"].tolist() == [1, 2, 3, 0, 4, 5, 0, 0]
    assert example["attention_spec"]["sequence_lengths"].tolist() == [4, 4]
    assert example["attention_spec"]["valid_lengths"].tolist() == [3, 2]


def test_estf_runs_paired_m2t_t2t_forwards_and_uses_provider_mask_token() -> None:
    model = RecordingModel()
    context = RecordingStepContext(model)
    trainer = _trainer(
        ESTFConfig(block_size=2, max_stages=2, confidence_threshold=1.0),
        special_tokens={"mask_token_id": 29, "pad_token_id": 31},
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "loss_mask": torch.ones((1, 4), dtype=torch.bool),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_spec": {
            "sequence_lengths": torch.tensor([[4]]),
            "valid_lengths": torch.tensor([[4]]),
        },
    }

    metrics = trainer.training_step(batch, context)

    assert len(model.inputs) == 4
    assert model.inputs[0].tolist() == [[29, 29, 29, 29, 1, 2, 3, 4]]
    assert (model.inputs[1][0, :4] == 0).sum().item() == 2
    assert (model.inputs[1][0, :4] == 29).sum().item() == 2
    stage_zero_known = model.inputs[1][0, :4] != 29
    assert torch.equal(model.inputs[2][0, :4][stage_zero_known], batch["input_ids"][0][stage_zero_known])
    assert torch.all(model.inputs[2][0, :4][~stage_zero_known] == 29)
    assert model.inputs[3].tolist() == [[0, 0, 0, 0, 1, 2, 3, 4]]

    assert len(context.backward_calls) == 4
    assert [weight for _, weight in context.backward_calls] == [0.25] * 4
    expected_loss = sum(loss.item() * weight for loss, weight in context.backward_calls)
    assert metrics.values["loss"] == pytest.approx(expected_loss)
    assert metrics.values["stage_0/forced_fill_tokens"] == 2
    assert metrics.values["stage_0/t2t_tokens"] == 2
    assert metrics.values["stage_1/m2t_tokens"] == 2
    assert metrics.values["stage_1/t2t_tokens"] == 4
    assert metrics.values["coverage"] == 1.0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_estf_selective_loss_limits_non_final_m2t_objective_to_new_tokens() -> None:
    model = RecordingModel()
    context = RecordingStepContext(model)
    trainer = _trainer(
        ESTFConfig(block_size=2, max_stages=2, confidence_threshold=1.0, selective_loss=True),
        special_tokens={"mask_token_id": 29, "pad_token_id": 31},
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "loss_mask": torch.ones((1, 4), dtype=torch.bool),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_spec": {
            "sequence_lengths": torch.tensor([[4]]),
            "valid_lengths": torch.tensor([[4]]),
        },
    }

    metrics = trainer.training_step(batch, context)

    assert metrics.values["stage_0/m2t_tokens"] == 2
    assert metrics.values["stage_1/m2t_tokens"] == 2
    assert metrics.values["stage_1/m2t_loss"] == pytest.approx(metrics.values["stage_0/m2t_loss"])
    assert len(context.backward_calls) == 4


def test_estf_random_known_initialization_preserves_each_block_end() -> None:
    model = RecordingModel()
    context = RecordingStepContext(model)
    trainer = _trainer(
        ESTFConfig(
            block_size=2,
            max_stages=2,
            confidence_threshold=1.0,
            initialize_random_known=True,
            initial_known_ratio=(1.0, 1.0),
            initial_known_schedule_power=2.0,
        ),
        special_tokens={"mask_token_id": 29, "pad_token_id": 31},
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "loss_mask": torch.ones((1, 4), dtype=torch.bool),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_spec": {
            "sequence_lengths": torch.tensor([[4]]),
            "valid_lengths": torch.tensor([[4]]),
        },
    }

    metrics = trainer.training_step(batch, context)

    assert model.inputs[0].tolist() == [[1, 29, 3, 29, 1, 2, 3, 4]]
    assert metrics.values["initial_known_tokens"] == 2


def test_recipe_spec_accepts_only_token_records_and_selects_trainer() -> None:
    recipe_spec.validate_run(
        recipe_config=ESTFConfig(block_size=4),
        run_config=RunConfig(data=DataConfig(format="token_records", max_seq_len=8)),
    )
    assert recipe_spec.trainer_type is ESTFTrainer

    with pytest.raises(ValueError, match="token_records only"):
        recipe_spec.validate_run(
            recipe_config=ESTFConfig(),
            run_config=RunConfig(data=DataConfig(format="conversation")),
        )


def test_public_estf_example_is_a_valid_run_config() -> None:
    project_root = Path(__file__).parents[3]
    config = load_config(project_root / "configs/recipes/estf/llada2_mini.yaml")

    assert config.recipe.name == "estf"
    assert config.recipe.params["max_stages"] == 4
    assert config.recipe.params["confidence_source"] == "maximum"
    assert config.recipe.params["initialize_random_known"] is True
    assert config.recipe.params["initial_known_ratio"] == [0.1, 0.6]
    assert config.recipe.params["initial_known_schedule_power"] == 2.0
    assert recipe_spec.config_type is ESTFConfig
    assert recipe_spec.trainer_type is ESTFTrainer
