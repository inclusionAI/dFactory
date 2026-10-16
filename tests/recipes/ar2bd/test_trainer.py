"""Behavior contract for the AR2BD trainer."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from dfactory.config import DataConfig, RunConfig
from dfactory.data.transforms.token_records import transform_token_record
from dfactory.recipes.ar2bd import AR2BDConfig, AR2BDTrainer, build_data, recipe_spec


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)
        self.forward_kwargs = None

    def forward(self, input_ids, **kwargs):
        self.forward_kwargs = dict(kwargs)
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


class PositionLogitModel(torch.nn.Module):
    def __init__(self, sequence_length: int) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, sequence_length, 32))

    def forward(self, input_ids, **kwargs):
        del kwargs
        return SimpleNamespace(logits=self.logits.expand(input_ids.shape[0], -1, -1))


class RecordingStepContext:
    global_step = 0
    epoch = 0
    accumulation_steps = 1
    device = "cpu"

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.model_inputs = None

    def forward(self, model_inputs):
        self.model_inputs = dict(model_inputs)
        return self.model(**self.model_inputs)

    def backward(self, loss, *, weight: float = 1.0) -> None:
        (loss * weight).backward()

    def reduce_loss(self, loss, token_count):
        del token_count
        return loss


def _batched(example: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.unsqueeze(0) for name, value in example.items()}


def _trainer(config: AR2BDConfig, *, special_tokens: dict[str, int]) -> AR2BDTrainer:
    trainer = AR2BDTrainer(
        run_config=RunConfig(),
        recipe_config=config,
        data_factory=build_data,
    )
    trainer.special_tokens = special_tokens
    return trainer


def test_ar2bd_config_is_strict_and_validates_algorithm_parameters() -> None:
    assert AR2BDConfig.from_mapping({}) == AR2BDConfig()
    assert AR2BDConfig.from_mapping({"noise_range": [0.0, 1.0]}).noise_range == (0.0, 1.0)
    with pytest.raises(ValueError, match="unknown AR2BD parameter"):
        AR2BDConfig.from_mapping({"doc_level": True})
    with pytest.raises(ValueError, match="must contain two values"):
        AR2BDConfig.from_mapping({"noise_range": 0.5})
    with pytest.raises(ValueError, match="0 <= low <= high <= 1"):
        AR2BDConfig.from_mapping({"noise_range": [0.8, 0.2]})


def test_token_record_transform_keeps_clean_tokens_and_padding_metadata() -> None:
    example = transform_token_record(
        {"tokens": [1, 2, 3]},
        max_seq_len=8,
        block_size=4,
        pad_token_id=0,
        token_key="tokens",
        pad_to_max_length=False,
    )

    assert example["input_ids"].tolist() == [1, 2, 3, 0]
    assert example["labels"].tolist() == [1, 2, 3, -100]
    assert example["loss_mask"].tolist() == [True, True, True, False]
    assert example["attention_spec"]["sequence_lengths"].tolist() == [4]
    assert example["attention_spec"]["valid_lengths"].tolist() == [3]
    assert "noisy_input_ids" not in example


def test_ar2bd_data_factory_packs_token_documents_without_losing_boundaries() -> None:
    recipe_data = build_data(
        source=[{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}],
        tokenizer=None,
        data_config=DataConfig(
            path="unused",
            format="token_records",
            text_key="input_ids",
            max_seq_len=8,
            packing=True,
        ),
        recipe_config=AR2BDConfig(block_size=4),
        special_tokens={"pad_token_id": 0},
    )

    example = recipe_data.dataset[0]
    assert example["input_ids"].tolist() == [1, 2, 3, 0, 4, 5, 6, 0]
    assert example["attention_spec"]["sequence_lengths"].tolist() == [4, 4]
    assert example["attention_spec"]["valid_lengths"].tolist() == [3, 3]
    assert example["labels"].tolist() == [1, 2, 3, -100, 4, 5, 6, -100]


def test_ar2bd_samples_runtime_noise_and_isolates_packed_documents() -> None:
    recipe_data = build_data(
        source=[{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}],
        tokenizer=None,
        data_config=DataConfig(
            path="unused",
            format="token_records",
            text_key="input_ids",
            max_seq_len=8,
            packing=True,
        ),
        recipe_config=AR2BDConfig(block_size=2),
        special_tokens={"pad_token_id": 0},
    )
    model = TinyModel()
    context = RecordingStepContext(model)
    trainer = _trainer(
        AR2BDConfig(block_size=2, noise_range=(1.0, 1.0), same_token_labels=True),
        special_tokens={"mask_token_id": 31, "pad_token_id": 0},
    )

    metrics = trainer.training_step(_batched(recipe_data.dataset[0]), context)

    assert context.model_inputs["input_ids"].tolist() == [[31, 31, 31, 0, 1, 2, 3, 0, 31, 31, 31, 0, 4, 5, 6, 0]]
    attention = context.model_inputs["attention_mask"]
    assert attention.shape == (1, 1, 16, 16)
    assert attention[0, 0, 1, 0] == 0
    assert torch.isneginf(attention[0, 0, 0, 8])
    assert torch.isneginf(attention[0, 0, 8, 0])
    assert torch.isneginf(attention[0, 0, 3]).all()
    assert metrics.values["loss_tokens"] == 6
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_ar2bd_shares_one_noise_rate_across_each_documents_blocks(monkeypatch) -> None:
    model = TinyModel()
    context = RecordingStepContext(model)
    trainer = _trainer(
        AR2BDConfig(block_size=2, noise_range=(0.0, 1.0), same_token_labels=True),
        special_tokens={"mask_token_id": 31, "pad_token_id": 0},
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "labels": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "loss_mask": torch.ones((1, 8), dtype=torch.bool),
        "position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
        "attention_spec": {
            "sequence_lengths": torch.tensor([[4, 4]]),
            "valid_lengths": torch.tensor([[4, 4]]),
        },
    }
    random_values = iter((torch.tensor([0.0, 1.0]), torch.full((1, 16), 0.5)))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: next(random_values))

    trainer.training_step(batch, context)

    assert context.model_inputs["input_ids"].tolist() == [[1, 2, 3, 4, 1, 2, 3, 4, 31, 31, 31, 31, 5, 6, 7, 8]]


def test_ar2bd_shifted_objective_uses_only_runtime_masked_labels() -> None:
    model = TinyModel()
    context = RecordingStepContext(model)
    trainer = _trainer(
        AR2BDConfig(block_size=2, noise_range=(1.0, 1.0), same_token_labels=False),
        special_tokens={"mask_token_id": 31, "pad_token_id": 0},
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

    assert metrics.values["loss_tokens"] == 3
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_ar2bd_shifted_objective_does_not_cross_packed_document_boundaries() -> None:
    recipe_data = build_data(
        source=[{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}],
        tokenizer=None,
        data_config=DataConfig(
            path="unused",
            format="token_records",
            text_key="input_ids",
            max_seq_len=8,
            packing=True,
        ),
        recipe_config=AR2BDConfig(block_size=2),
        special_tokens={"pad_token_id": 0},
    )
    model = PositionLogitModel(sequence_length=16)
    context = RecordingStepContext(model)
    trainer = _trainer(
        AR2BDConfig(block_size=2, noise_range=(1.0, 1.0), same_token_labels=False),
        special_tokens={"mask_token_id": 31, "pad_token_id": 0},
    )

    metrics = trainer.training_step(_batched(recipe_data.dataset[0]), context)

    assert metrics.values["loss_tokens"] == 4
    assert not model.logits.grad[0, 7].any()
    assert model.logits.grad[0, 8].any()


def test_recipe_spec_accepts_only_token_records_and_selects_trainer() -> None:
    config = RunConfig(data=DataConfig(format="token_records", max_seq_len=8))
    recipe_spec.validate_run(recipe_config=AR2BDConfig(block_size=4), run_config=config)
    assert recipe_spec.trainer_type is AR2BDTrainer

    with pytest.raises(ValueError, match="token_records only"):
        recipe_spec.validate_run(
            recipe_config=AR2BDConfig(),
            run_config=RunConfig(data=DataConfig(format="conversation")),
        )


def test_public_ar2bd_example_records_the_complete_recipe_contract() -> None:
    project_root = Path(__file__).parents[3]
    raw = yaml.safe_load((project_root / "configs/recipes/ar2bd/llada2_mini.yaml").read_text())

    assert raw["recipe"]["name"] == "ar2bd"
    assert AR2BDConfig.from_mapping(raw["recipe"]["params"]).same_token_labels is True
    assert raw["data"]["format"] == "token_records"
    assert raw["data"]["packing"] is True
