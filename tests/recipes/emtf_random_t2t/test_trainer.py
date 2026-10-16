"""Behavior tests for the built-in EMTF random-T2T trainer."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dfactory.config import DataConfig, RunConfig, load_config
from dfactory.recipes.emtf_random_t2t import (
    EMTFRandomT2TConfig,
    EMTFRandomT2TTrainer,
    build_data,
    recipe_spec,
)


class MaskAwareTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        input_ids = list(range(1, len(messages) * 2 + 1))
        assistant_mask = []
        for message in messages:
            assistant_mask.extend([message["role"] == "assistant"] * 2)
        return {"input_ids": input_ids, "assistant_masks": assistant_mask}


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
    global_step = 3
    epoch = 0
    accumulation_steps = 4
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


def _conversation() -> dict[str, list[dict[str, str]]]:
    return {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }


def _trainer(config: EMTFRandomT2TConfig, *, special_tokens: dict[str, int]) -> EMTFRandomT2TTrainer:
    trainer = EMTFRandomT2TTrainer(
        run_config=RunConfig(),
        recipe_config=config,
        data_factory=build_data,
    )
    trainer.special_tokens = special_tokens
    return trainer


def test_random_t2t_config_is_strict() -> None:
    assert EMTFRandomT2TConfig.from_mapping({"max_stages": 3}).max_stages == 3
    with pytest.raises(ValueError, match="unknown EMTF random-T2T parameter"):
        EMTFRandomT2TConfig.from_mapping({"per_stage_step": True})
    with pytest.raises(ValueError, match="max_stages must be at least 2"):
        EMTFRandomT2TConfig.from_mapping({"max_stages": 1})


def test_random_t2t_data_factory_uses_the_conversation_contract() -> None:
    recipe_data = build_data(
        source=[_conversation(), _conversation()],
        tokenizer=MaskAwareTokenizer(),
        data_config=DataConfig(path="unused", max_seq_len=8, packing=True),
        recipe_config=EMTFRandomT2TConfig(block_size=2),
        special_tokens={"pad_token_id": 0},
    )

    example = recipe_data.dataset[0]
    assert example["input_ids"].shape == (8,)
    assert example["attention_spec"]["sequence_lengths"].tolist() == [4, 4]
    assert example["loss_mask"].sum().item() == 4


def test_random_t2t_runs_one_uniformly_selected_sc_with_source_weight(monkeypatch) -> None:
    randint_call = None

    def choose_stage(low, high, size, *, device):
        nonlocal randint_call
        randint_call = (low, high, size, device)
        return torch.tensor([1], device=device)

    monkeypatch.setattr(torch, "randint", choose_stage)
    model = RecordingModel()
    context = RecordingStepContext(model)
    trainer = _trainer(
        EMTFRandomT2TConfig(block_size=3, max_stages=3, confidence_threshold=1.0),
        special_tokens={"mask_token_id": 31, "eos_token_id": 30, "pad_token_id": 0},
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "labels": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "loss_mask": torch.ones((1, 6), dtype=torch.bool),
        "position_ids": torch.arange(6).unsqueeze(0),
        "attention_spec": {
            "sequence_lengths": torch.tensor([[6]]),
            "valid_lengths": torch.tensor([[6]]),
        },
    }

    metrics = trainer.training_step(batch, context)

    assert randint_call == (0, 3, (1,), batch["input_ids"].device)
    assert len(model.inputs) == 4
    assert model.inputs[0][0, :6].tolist() == [31] * 6
    assert (model.inputs[1][0, :6] != 31).sum().item() == 2
    assert (model.inputs[2][0, :6] != 31).sum().item() == 4
    assert torch.all(model.inputs[2][0, :6][model.inputs[2][0, :6] != 31] == 0)
    assert (model.inputs[3][0, :6] != 31).sum().item() == 4

    assert len(context.backward_calls) == 4
    assert [weight for _, weight in context.backward_calls] == pytest.approx([1 / 6, 1 / 6, 1 / 3, 1 / 6])
    expected_loss = sum(loss.item() * weight for loss, weight in context.backward_calls)
    assert metrics.values["loss"] == pytest.approx(expected_loss)
    assert metrics.values["sc_stage"] == 1
    assert metrics.values["stage_0/sc_loss"] == 0
    assert metrics.values["stage_0/sc_tokens"] == 0
    assert metrics.values["stage_1/sc_tokens"] == 4
    assert metrics.values["stage_2/sc_loss"] == 0
    assert metrics.values["coverage"] == 1.0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_public_random_t2t_example_is_a_valid_run_config() -> None:
    project_root = Path(__file__).parents[3]
    config = load_config(project_root / "configs/recipes/emtf_random_t2t/llada2_mini.yaml")

    assert config.recipe.name == "emtf_random_t2t"
    assert config.recipe.params["max_stages"] == 4
    assert recipe_spec.config_type is EMTFRandomT2TConfig
    assert recipe_spec.trainer_type is EMTFRandomT2TTrainer
