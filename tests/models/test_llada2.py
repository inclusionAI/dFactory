import json
from pathlib import Path

import pytest
import torch

import dfactory.models.llada2.provider as provider_module
from dfactory.components.block_diffusion import (
    SlidingWindowAttentionConfig,
    build_attention_mask,
    build_block_diffusion_attention_masks,
    build_paired_layout,
)
from dfactory.config import ModelConfig
from dfactory.models.llada2 import LLaDA2MoeConfig, LLaDA2MoeModelLM
from dfactory.models.llada2.modeling import LLaDA2MoeExperts, LLaDA2MoeSparseMoeBlock


@pytest.mark.parametrize("model_name", ["llada2_mini", "llada2_flash"])
def test_bundled_model_config_loads_through_dfactory(model_name: str) -> None:
    config_path = Path(__file__).parents[2] / "configs" / "model_configs" / model_name

    config = LLaDA2MoeConfig.from_pretrained(config_path)

    assert config.model_type == "llada2_moe_veomni"
    assert not hasattr(config, "auto_map")


def test_hugging_face_export_round_trip_uses_dfactory_model_class(tmp_path: Path) -> None:
    config = LLaDA2MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        first_k_dense_replace=1,
        max_position_embeddings=64,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = LLaDA2MoeModelLM(config)

    class Tokenizer:
        def save_pretrained(self, path) -> None:
            (Path(path) / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")

    provider_module.LLaDA2Provider().save_weights(model, Tokenizer(), model.state_dict(), tmp_path)
    restored = LLaDA2MoeModelLM.from_pretrained(tmp_path)

    restored_state = restored.state_dict()
    assert set(model.state_dict()) == set(restored_state)
    assert all(torch.equal(value, restored_state[name]) for name, value in model.state_dict().items())
    exported_config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert exported_config["auto_map"] == {
        "AutoConfig": "configuration_dfactory_llada2.LLaDA2MoeConfig",
        "AutoModel": "modeling_dfactory_llada2.LLaDA2MoeModel",
        "AutoModelForCausalLM": "modeling_dfactory_llada2.LLaDA2MoeModelLM",
    }
    assert (tmp_path / "configuration_dfactory_llada2.py").is_file()
    assert (tmp_path / "modeling_dfactory_llada2.py").is_file()


def test_provider_applies_explicit_special_tokens_to_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    model_config = LLaDA2MoeConfig(pad_token_id=1, eos_token_id=2)
    monkeypatch.setattr(provider_module.LLaDA2MoeConfig, "from_pretrained", lambda path: model_config)
    monkeypatch.setattr(provider_module, "LLaDA2MoeModelLM", lambda config: type("Model", (), {"config": config})())
    config = ModelConfig(
        config_path="config",
        weights_path="weights",
        tokenizer_path="tokenizer",
        mask_token_id=11,
        eos_token_id=12,
        pad_token_id=13,
    )

    model = provider_module.LLaDA2Provider().build_model(config)

    assert model.config.mask_token_id == 11
    assert model.config.eos_token_id == 12
    assert model.config.pad_token_id == 13
    assert model.config._moe_implementation == "fused"


@pytest.mark.parametrize(
    ("implementation", "expert_type"),
    [("fused", LLaDA2MoeExperts), ("eager", torch.nn.ModuleList)],
)
def test_sparse_moe_constructs_the_selected_expert_implementation(implementation, expert_type) -> None:
    config = LLaDA2MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=1,
        n_group=2,
        topk_group=1,
    )
    config._moe_implementation = implementation

    block = LLaDA2MoeSparseMoeBlock(config)

    assert isinstance(block.experts, expert_type)


def test_tiny_llada2_runs_with_transformers_457() -> None:
    config = LLaDA2MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        first_k_dense_replace=1,
        max_position_embeddings=64,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = LLaDA2MoeModelLM(config)
    layout = build_paired_layout([[4]], block_size=2)
    allowed = build_attention_mask(layout)
    attention_mask = torch.zeros((1, 1, 8, 8)).masked_fill_(~allowed.unsqueeze(1), float("-inf"))

    result = model(
        input_ids=torch.arange(8).unsqueeze(0),
        position_ids=layout.position_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    assert result.logits.shape == (1, 8, 64)


def test_llada2_selects_recipe_attention_mask_per_layer() -> None:
    config = LLaDA2MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        first_k_dense_replace=2,
        max_position_embeddings=64,
        pad_token_id=0,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = LLaDA2MoeModelLM(config)
    layout = build_paired_layout([[6]], block_size=2)
    masks = build_block_diffusion_attention_masks(
        layout,
        backend="sdpa",
        sliding_window=SlidingWindowAttentionConfig(window_blocks=(1, None), sink_tokens=1),
    )
    seen_masks = []

    def capture_mask(_module, _args, kwargs):
        seen_masks.append(kwargs["attention_mask"])

    handles = [layer.register_forward_pre_hook(capture_mask, with_kwargs=True) for layer in model.model.layers]
    try:
        model(
            input_ids=torch.arange(12).unsqueeze(0),
            position_ids=layout.position_ids,
            attention_mask=masks.base,
            block_diffusion_attention_masks=masks,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert seen_masks[0] is masks.sliding_windows[1]
    assert torch.equal(seen_masks[1], masks.base)
