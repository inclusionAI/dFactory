"""Behavior tests for the built-in EMTF trainer."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dfactory.config import DataConfig, RunConfig, load_config
from dfactory.recipes.emtf import EMTFConfig, EMTFTrainer, build_data, recipe_spec


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


def _trainer(config: EMTFConfig, *, special_tokens: dict[str, int]) -> EMTFTrainer:
    trainer = EMTFTrainer(
        run_config=RunConfig(),
        recipe_config=config,
        data_factory=build_data,
    )
    trainer.special_tokens = special_tokens
    return trainer


@pytest.mark.parametrize("legacy_parameter", ["per_stage_step", "enable_swa", "same_token_labels"])
def test_emtf_config_rejects_old_inherited_parameters(legacy_parameter: str) -> None:
    with pytest.raises(ValueError, match="unknown EMTF parameter"):
        EMTFConfig.from_mapping({legacy_parameter: True})


def test_emtf_config_validates_owned_parameters() -> None:
    assert EMTFConfig.from_mapping({"max_stages": 2}).max_stages == 2
    with pytest.raises(ValueError, match="max_stages must be positive"):
        EMTFConfig.from_mapping({"max_stages": 0})
    with pytest.raises(ValueError, match="force_mask_token_ids must be a sequence"):
        EMTFConfig.from_mapping({"force_mask_token_ids": 7})


def test_emtf_data_factory_reuses_the_conversation_contract() -> None:
    recipe_data = build_data(
        source=[_conversation(), _conversation()],
        tokenizer=MaskAwareTokenizer(),
        data_config=DataConfig(path="unused", max_seq_len=8, packing=True),
        recipe_config=EMTFConfig(block_size=2),
        special_tokens={"pad_token_id": 0},
    )

    example = recipe_data.dataset[0]
    assert example["input_ids"].shape == (8,)
    assert example["attention_spec"]["sequence_lengths"].tolist() == [4, 4]
    assert example["loss_mask"].sum().item() == 4


def test_emtf_runs_teacher_forcing_and_self_correction_for_every_stage() -> None:
    model = RecordingModel()
    context = RecordingStepContext(model)
    trainer = _trainer(
        EMTFConfig(block_size=2, max_stages=2, confidence_threshold=1.0),
        special_tokens={"mask_token_id": 31, "eos_token_id": 30, "pad_token_id": 0},
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
    assert model.inputs[0][0, :4].tolist() == [31, 31, 31, 31]
    first_selection = model.inputs[1][0, :4] != 31
    assert first_selection.sum().item() == 2
    assert torch.all(model.inputs[1][0, :4][first_selection] == 0)
    assert torch.equal(model.inputs[2][0, :4][first_selection], batch["input_ids"][0][first_selection])
    assert model.inputs[3][0, :4].tolist() == [0, 0, 0, 0]

    assert len(context.backward_calls) == 4
    assert [weight for _, weight in context.backward_calls] == [0.25] * 4
    expected_loss = sum(loss.item() * weight for loss, weight in context.backward_calls)
    assert metrics.values["loss"] == pytest.approx(expected_loss)
    assert metrics.values["loss"] == pytest.approx(metrics.values["tf_loss"] + metrics.values["sc_loss"])
    assert metrics.values["stage_0/tf_tokens"] == 2
    assert metrics.values["stage_0/sc_tokens"] == 2
    assert metrics.values["stage_0/error_rate"] == 1.0
    assert metrics.values["stage_0/sc_error_rate"] == 1.0
    assert metrics.values["stage_1/tf_tokens"] == 2
    assert metrics.values["stage_1/sc_tokens"] == 4
    assert metrics.values["coverage"] == 1.0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_public_emtf_example_is_a_valid_run_config() -> None:
    project_root = Path(__file__).parents[3]
    config = load_config(project_root / "configs/recipes/emtf/llada2_mini.yaml")

    assert config.recipe.name == "emtf"
    assert config.recipe.params["max_stages"] == 4
    assert recipe_spec.config_type is EMTFConfig
    assert recipe_spec.trainer_type is EMTFTrainer
