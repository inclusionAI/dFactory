"""Behavior tests for the built-in SFT trainer."""

from types import SimpleNamespace

import torch

from dfactory.components.block_diffusion import SlidingWindowAttentionConfig
from dfactory.config import ModelConfig, RunConfig
from dfactory.recipes.sft import SFTConfig, SFTTrainer, build_data
from dfactory.trainers import StepContext, TrainerState


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16)
        self.forward_kwargs = None

    def forward(self, input_ids, **kwargs):
        self.forward_kwargs = kwargs
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def _trainer(config: SFTConfig, *, special_tokens: dict[str, int]) -> SFTTrainer:
    trainer = SFTTrainer(
        run_config=RunConfig(model=ModelConfig(attention="sdpa")),
        recipe_config=config,
        data_factory=build_data,
    )
    trainer.special_tokens = special_tokens
    return trainer


def test_sft_trainer_runs_noise_forward_and_backward() -> None:
    model = TinyModel()
    trainer = _trainer(
        SFTConfig(block_size=2, noise_range=(1.0, 1.0)),
        special_tokens={"mask_token_id": 15, "eos_token_id": 14},
    )
    context = StepContext(
        model=model,
        state=TrainerState(),
        accumulation_steps=1,
        device="cpu",
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, 2, 3, -100]]),
        "loss_mask": torch.tensor([[False, True, True, False]]),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_spec": {
            "sequence_lengths": torch.tensor([[4]]),
            "valid_lengths": torch.tensor([[4]]),
        },
    }

    metrics = trainer.training_step(batch, context)

    assert metrics.values["loss"] > 0
    assert metrics.values["loss_tokens"] == 3
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_sft_trainer_passes_recipe_owned_sliding_window_masks() -> None:
    model = TinyModel()
    trainer = _trainer(
        SFTConfig(
            block_size=2,
            noise_range=(1.0, 1.0),
            sliding_window=SlidingWindowAttentionConfig(window_blocks=1, sink_tokens=1),
        ),
        special_tokens={"mask_token_id": 15, "eos_token_id": 14},
    )
    context = StepContext(
        model=model,
        state=TrainerState(global_step=7),
        accumulation_steps=1,
        device="cpu",
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[-100, 2, 3, -100]]),
        "loss_mask": torch.tensor([[False, True, True, False]]),
        "position_ids": torch.arange(4).unsqueeze(0),
        "attention_spec": {
            "sequence_lengths": torch.tensor([[4]]),
            "valid_lengths": torch.tensor([[4]]),
        },
    }

    trainer.training_step(batch, context)

    assert model.forward_kwargs["training_step"] == 7
    masks = model.forward_kwargs["block_diffusion_attention_masks"]
    assert masks.for_layer(0, training_step=7) is masks.sliding_windows[1]


def test_sft_config_parses_nested_sliding_window_settings() -> None:
    config = SFTConfig.from_mapping(
        {
            "block_size": 32,
            "sliding_window": {
                "window_blocks": [16, None],
                "sink_tokens": 4,
                "activation_order": [0, 1],
                "activation_interval": 500,
            },
        }
    )

    assert config.sliding_window is not None
    assert config.sliding_window.window_blocks == (16, None)
    assert config.sliding_window.activation_order == (0, 1)
