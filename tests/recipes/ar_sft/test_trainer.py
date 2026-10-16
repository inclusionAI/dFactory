"""Behavior tests for the built-in autoregressive SFT trainer."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dfactory.config import DataConfig, RunConfig, load_config
from dfactory.recipes.ar_sft import ARSFTConfig, ARSFTTrainer, build_data, recipe_spec
from dfactory.recipes.ar_sft.attention import (
    build_attention_mask,
    build_document_layout,
    build_flex_attention_mask,
)


class MaskAwareTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        input_ids = list(range(1, len(messages) * 2 + 1))
        assistant_mask = []
        for message in messages:
            assistant_mask.extend([message["role"] == "assistant"] * 2)
        return {"input_ids": input_ids, "assistant_masks": assistant_mask}


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16)

    def forward(self, input_ids, **kwargs):
        del kwargs
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


class RecordingStepContext:
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


def _conversation() -> dict[str, list[dict[str, str]]]:
    return {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }


def _trainer(config: ARSFTConfig) -> ARSFTTrainer:
    return ARSFTTrainer(
        run_config=RunConfig(),
        recipe_config=config,
        data_factory=build_data,
    )


def test_ar_sft_config_is_strict_and_has_no_block_diffusion_parameters() -> None:
    assert ARSFTConfig.from_mapping({}) == ARSFTConfig()
    with pytest.raises(ValueError, match="unknown AR SFT parameter"):
        ARSFTConfig.from_mapping({"block_size": 32})


def test_ar_sft_data_factory_packs_documents_without_interleaving() -> None:
    recipe_data = build_data(
        source=[_conversation(), _conversation()],
        tokenizer=MaskAwareTokenizer(),
        data_config=DataConfig(path="unused", max_seq_len=8, packing=True),
        recipe_config=ARSFTConfig(),
        special_tokens={"pad_token_id": 0},
    )

    example = recipe_data.dataset[0]
    assert example["input_ids"].shape == (8,)
    assert example["attention_spec"]["sequence_lengths"].tolist() == [4, 4]
    assert example["attention_spec"]["valid_lengths"].tolist() == [4, 4]
    assert "noisy_input_ids" not in example


def test_document_attention_is_causal_isolated_and_hides_padding() -> None:
    layout = build_document_layout([[2, 2]], total_length=6, valid_lengths=[[2, 1]])
    mask = build_attention_mask(layout)[0]

    assert layout.document_ids.tolist() == [[1, 1, 2, 0, 0, 0]]
    assert layout.position_ids.tolist() == [[0, 1, 0, 1, 0, 0]]
    assert mask[1, 0]
    assert not mask[0, 1]
    assert not mask[:2, 2:4].any()
    assert not mask[2:4, :2].any()
    assert not mask[3:].any()
    assert not mask[:, 3:].any()


def test_flex_attention_preserves_document_causal_semantics() -> None:
    layout = build_document_layout([[128, 128]])
    block_mask = build_flex_attention_mask(layout)

    assert block_mask.shape == (1, 1, 256, 256)
    assert block_mask.mask_mod(0, 0, 127, 0)
    assert not block_mask.mask_mod(0, 0, 0, 127)
    assert block_mask.mask_mod(0, 0, 128, 128)
    assert not block_mask.mask_mod(0, 0, 128, 127)


def test_ar_sft_runs_shifted_loss_without_noise_or_pair_duplication() -> None:
    model = TinyModel()
    context = RecordingStepContext(model)
    trainer = _trainer(ARSFTConfig())
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, 2, -100, 4]]),
        "loss_mask": torch.tensor([[False, True, False, True]]),
        "position_ids": torch.tensor([[0, 1, 0, 1]]),
        "attention_spec": {
            "sequence_lengths": torch.tensor([[2, 2]]),
            "valid_lengths": torch.tensor([[2, 2]]),
        },
    }

    metrics = trainer.training_step(batch, context)

    assert context.model_inputs["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert context.model_inputs["position_ids"].tolist() == [[0, 1, 0, 1]]
    attention_mask = context.model_inputs["attention_mask"]
    assert attention_mask.shape == (1, 1, 4, 4)
    assert torch.isneginf(attention_mask[0, 0, 2, 1])
    assert metrics.values["loss"] > 0
    assert metrics.values["loss_tokens"] == 2
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_recipe_spec_validates_only_ar_sft_owned_constraints() -> None:
    config = RunConfig(data=DataConfig(format="conversation", max_seq_len=7))
    recipe_spec.validate_run(recipe_config=ARSFTConfig(), run_config=config)
    assert recipe_spec.trainer_type is ARSFTTrainer


def test_public_ar_sft_example_is_a_valid_run_config() -> None:
    project_root = Path(__file__).parents[3]
    config = load_config(project_root / "configs/recipes/ar_sft/llada2_mini.yaml")

    assert config.recipe.name == "ar_sft"
    assert config.recipe.params == {}
    assert config.data.packing is True
