"""Qwen3 model behavior, head initialization and checkpoint loading."""

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config

import dfactory.models.qwen3.modeling as modeling
from dfactory.config import ModelConfig, load_config
from dfactory.models.qwen3 import Qwen3ForBDLM, Qwen3Provider
from dfactory.recipes.soldiff.artifacts import (
    load_p_head_weights,
    load_q_model,
    prepare_p_head,
    save_q_weights,
    save_recipe_assets,
)
from dfactory.recipes.soldiff.p_head import TokenScoreMLP, get_p_head
from dfactory.recipes.soldiff.q_model import build_q_model, load_q_config


@pytest.fixture(autouse=True)
def single_process_model_math(monkeypatch):
    # Model math does not require initializing a distributed process group.
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))


def tiny_config(**overrides):
    values = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=None,
        eos_token_id=2,
        attention_dropout=0.0,
        use_cache=False,
        tie_word_embeddings=True,
    )
    values.update(overrides)
    config = Qwen3Config(**values)
    config._attn_implementation = "eager"
    return config


def q_config_file(tmp_path, **overrides):
    config = dict(
        format="mask_generator_q_linear.v1",
        base_model_type="qwen3",
        encoder_num_layers=1,
        hidden_size=32,
        rms_eps=1e-6,
        q_score_bias=True,
        q_head_type="linear",
        base_config=tiny_config().to_dict(),
    )
    config.update(overrides)
    path = tmp_path / "q_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class Tokenizer:
    pad_token_id = 17
    eos_token_id = 9

    def save_pretrained(self, path):
        (Path(path) / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


def test_fresh_p_head_derives_width_and_initialization_from_backbone(tmp_path):
    model = Qwen3ForBDLM(tiny_config(initializer_range=0.013))
    path = tmp_path / "p_head_config.json"
    path.write_text(
        json.dumps(
            {
                "hidden_size": 999,
                "initializer_std": 9.0,
                "intermediate_size": 12,
                "rms_eps": 3e-6,
                "activation": "silu",
            }
        ),
        encoding="utf-8",
    )
    resolved = prepare_p_head(model, p_head_config_path=path)
    # Head config customizes intermediate_size/rms_eps/activation;
    # backbone width and initializer_range determine the other fields.
    assert resolved["hidden_size"] == 32
    assert resolved["initializer_std"] == 0.013
    assert resolved["intermediate_size"] == 12
    assert resolved["rms_eps"] == 3e-6


def test_provider_keeps_native_pad_none_and_tokenizer_eos_without_resizing(tmp_path):
    native = tiny_config()
    native.save_pretrained(tmp_path)
    config = ModelConfig(
        config_path=str(tmp_path),
        attention="eager",
        mask_token_id=151669,
        eos_token_id=151645,
        pad_token_id=151643,
    )
    provider = Qwen3Provider()
    model = provider.build_model(config)
    tokens = provider.special_tokens(Tokenizer(), model.config, config)

    assert type(model.config) is Qwen3Config
    assert model.config.pad_token_id is None
    assert model.config.eos_token_id == native.eos_token_id
    assert model.get_input_embeddings().padding_idx is None
    assert model.get_input_embeddings().weight.is_meta
    assert not model.model.rotary_emb.inv_freq.is_meta
    assert model.config.vocab_size == native.vocab_size
    assert model.get_input_embeddings().weight.shape[0] == native.vocab_size
    assert tokens == {"mask_token_id": 151669, "eos_token_id": 151645, "pad_token_id": 151643}
    assert provider.special_tokens(Tokenizer(), model.config, ModelConfig()) == {
        "mask_token_id": 151669,
        "eos_token_id": 9,
        "pad_token_id": 17,
    }
    assert model.config.pad_token_id is None
    assert model.get_input_embeddings().padding_idx is None
    model.config.pad_token_id = 11
    assert provider.special_tokens(Tokenizer(), model.config, ModelConfig())["pad_token_id"] == 17
    assert provider.special_tokens(SimpleNamespace(eos_token_id=9), model.config, ModelConfig())["pad_token_id"] == 11


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_external_mask_controls_future_token_outputs_and_gradients(attention):
    torch.manual_seed(13)
    config = tiny_config()
    config._attn_implementation = attention
    model = Qwen3ForBDLM(config).eval()
    inputs = torch.tensor([[4, 5, 6, 7]])
    changed = inputs.clone()
    changed[0, -1] = 8
    all_allowed = torch.zeros(1, 1, 4, 4)
    self_only = all_allowed.masked_fill(~torch.eye(4, dtype=torch.bool)[None, None], float("-inf"))

    def first_logits(ids, mask):
        return model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, 0]

    assert not torch.allclose(first_logits(inputs, all_allowed), first_logits(changed, all_allowed), atol=1e-7)
    torch.testing.assert_close(first_logits(inputs, self_only), first_logits(changed, self_only), atol=0, rtol=0)
    first_logits(inputs, all_allowed).square().sum().backward()
    full_grad = model.model.embed_tokens.weight.grad[7].norm()
    model.zero_grad(set_to_none=True)
    # Direct embeddings separate the input path from tied LM-head gradients.
    embeds = model.model.embed_tokens(inputs).detach().requires_grad_(True)
    model(inputs_embeds=embeds, attention_mask=self_only, use_cache=False).logits[:, 0].square().sum().backward()
    assert full_grad > 0
    assert torch.count_nonzero(embeds.grad[:, 1:]) == 0
    assert all(not layer.self_attn.is_causal for layer in model.model.layers)


def _mark_head_parameters(head):
    # Distinguish saved values from every random/default initializer, including
    # RMSNorm's all-ones default, then retain exact expected checkpoint tensors.
    with torch.no_grad():
        for index, parameter in enumerate(head.parameters()):
            values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
            parameter.copy_(values / parameter.numel() * 0.01 + (index + 1) * 0.125)
    return {name: value.clone() for name, value in head.state_dict().items()}


def test_main_checkpoint_round_trip_restores_native_config_bd_class_and_p_head(tmp_path):
    torch.manual_seed(17)
    model = Qwen3ForBDLM(tiny_config()).eval()
    prepare_p_head(model)
    expected_head = _mark_head_parameters(model.p_head)
    assert TokenScoreMLP.__name__ in model._no_split_modules
    inputs = torch.tensor([[3, 4, 5, 6]])
    mask = torch.zeros(1, 1, 4, 4)
    before = model(input_ids=inputs, attention_mask=mask, output_hidden_states=True, use_cache=False)
    before_scores = model.p_head(before.hidden_states[-1])
    provider = Qwen3Provider()
    provider.save_weights(model, Tokenizer(), model.state_dict(), tmp_path)

    config = AutoConfig.from_pretrained(tmp_path)
    assert type(config) is Qwen3Config
    assert config.pad_token_id is None
    assert config.p_head_config == model.config.p_head_config
    assert "AutoConfig" not in config.auto_map
    assert config.auto_map["AutoModelForCausalLM"] == "modeling_soldiff.Qwen3ForBDLM"
    assert (tmp_path / "modeling_soldiff.py").is_file()
    torch.manual_seed(93)
    restored, loading_info = AutoModelForCausalLM.from_pretrained(
        tmp_path,
        trust_remote_code=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    restored.eval()
    assert type(restored).__name__ == "Qwen3ForBDLM"
    assert type(restored).__module__.startswith("transformers_modules.")
    assert loading_info["missing_keys"] == loading_info["unexpected_keys"] == []
    head_id = id(restored.p_head)
    parameter_ids = {name: id(value) for name, value in restored.p_head.named_parameters()}
    rng = torch.get_rng_state().clone()
    prepare_p_head(restored)
    prepare_p_head(restored)
    assert id(restored.p_head) == head_id
    assert parameter_ids == {name: id(value) for name, value in restored.p_head.named_parameters()}
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for name, value in expected_head.items():
        torch.testing.assert_close(restored.p_head.state_dict()[name], value, rtol=0, atol=0)
    after = restored(input_ids=inputs, attention_mask=mask, output_hidden_states=True, use_cache=False)
    torch.testing.assert_close(before.logits, after.logits, atol=0, rtol=0)
    torch.testing.assert_close(before_scores, restored.p_head(after.hidden_states[-1]), atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    assert restored.lm_head.weight is restored.model.embed_tokens.weight


def test_training_hf_loader_restores_embedded_head_without_resetting_it(tmp_path):
    from veomni.models.module_utils import load_model_weights

    model = Qwen3ForBDLM(tiny_config())
    prepare_p_head(model)
    expected = _mark_head_parameters(model.p_head)
    provider = Qwen3Provider()
    provider.save_weights(model, Tokenizer(), model.state_dict(), tmp_path)

    restored = provider.build_model(ModelConfig(config_path=str(tmp_path), attention="eager"))
    head_id = id(restored.p_head)
    prepare_p_head(restored)
    assert id(restored.p_head) == head_id
    assert next(restored.p_head.parameters()).is_meta
    load_model_weights(restored, str(tmp_path), init_device="cpu")
    load_p_head_weights(restored, model_path=str(tmp_path))  # No standalone sidecar.
    materialized_head_id = id(restored.p_head)
    prepare_p_head(restored)
    assert id(restored.p_head) == materialized_head_id
    for name, value in expected.items():
        torch.testing.assert_close(restored.p_head.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("changed", [None, "rms_eps", "activation", "intermediate_size", "missing_metadata"])
def test_full_resume_checks_saved_head_structure_before_loading_tensors(tmp_path, changed):
    from dataclasses import replace

    from dfactory.config import RunConfig
    from dfactory.recipes.soldiff.config import SolDiffConfig
    from dfactory.recipes.soldiff.trainer import SolDiffTrainer

    saved = prepare_p_head(Qwen3ForBDLM(tiny_config()))
    if changed not in {None, "missing_metadata"}:
        saved[changed] = {"rms_eps": 2e-5, "activation": "relu", "intermediate_size": 8}[changed]
    path = tmp_path / "model_assets" / "p_head_config.json"
    if changed != "missing_metadata":
        path.parent.mkdir()
        path.write_text(json.dumps(saved), encoding="utf-8")
    run = RunConfig()
    run = replace(run, checkpoint=replace(run.checkpoint, resume_from=str(tmp_path)))
    trainer = SolDiffTrainer(
        run_config=run, recipe_config=SolDiffConfig(q_config_path="unused.json"), data_factory=lambda **_: None
    )
    model = Qwen3ForBDLM(tiny_config())
    if changed is None:
        assert trainer.prepare_main_model(model, None) is model
    else:
        message = "readable p head configuration" if changed == "missing_metadata" else changed
        with pytest.raises(ValueError, match=message):
            trainer.prepare_main_model(model, None)


@pytest.mark.parametrize("missing_name", ["norm.weight", "gate_proj.weight"])
def test_hf_missing_head_parameter_uses_head_initializer_without_resetting_loaded_values(tmp_path, missing_name):
    from safetensors.torch import load_file, save_file

    from dfactory.models.qwen3.heads import attach_p_head

    model = Qwen3ForBDLM(tiny_config())
    attach_p_head(model, config_values={"initializer_std": 0.15})
    expected = _mark_head_parameters(model.p_head)
    Qwen3Provider().save_weights(model, Tokenizer(), model.state_dict(), tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    del state[f"p_head.{missing_name}"]
    save_file(state, path)
    torch.manual_seed(149)
    restored, loading_info = AutoModelForCausalLM.from_pretrained(
        tmp_path, trust_remote_code=True, attn_implementation="eager", output_loading_info=True
    )
    assert f"p_head.{missing_name}" in loading_info["missing_keys"]
    for name, value in restored.p_head.state_dict().items():
        if name != missing_name:
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        elif name == "norm.weight":
            torch.testing.assert_close(value, torch.ones_like(value), rtol=0, atol=0)
        else:
            assert torch.isfinite(value).all()
            assert abs(float(value.std()) - 0.15) < 0.025


def test_head_meta_materialization_uses_its_own_initializer():
    with torch.device("meta"):
        model = Qwen3ForBDLM(tiny_config())
        prepare_p_head(model)
    head = get_p_head(model)
    assert next(head.parameters()).is_meta
    head.to_empty(device="cpu")
    # This is the initializer that missing-parameter loading must select.
    for parameter in head.parameters():
        parameter.data.fill_(float("nan"))
    from veomni.models.module_utils import _init_parameter

    for name, _parameter in head.named_parameters():
        _init_parameter(model, f"p_head.{name}")
    assert all(torch.isfinite(parameter).all() for parameter in head.parameters())
    torch.testing.assert_close(head.norm.weight, torch.ones_like(head.norm.weight))
    assert head.gate_proj.weight.std() > 0
    assert abs(float(head.gate_proj.weight.std()) - head.initializer_std) < 0.005


def test_separate_p_head_weights_are_loaded_after_materialization(tmp_path):
    from safetensors.torch import save_file

    model = Qwen3ForBDLM(tiny_config())
    prepare_p_head(model)
    expected = {f"p_head.{name}": torch.full_like(value, 0.25) for name, value in model.p_head.state_dict().items()}
    save_file(expected, tmp_path / "p_head.safetensors")
    load_p_head_weights(model, model_path=str(tmp_path))
    assert all(torch.all(value == 0.25) for value in model.p_head.state_dict().values())


def test_q_random_initialization_and_rope_meta_materialization(tmp_path):
    path = q_config_file(tmp_path)
    q = build_q_model(path, init_device="meta", attn_implementation="eager")
    assert next(q.parameters()).is_meta
    assert len(q.prefix_encoder.layers) == 1
    assert not hasattr(q, "lm_head")
    q.to_empty(device="cpu")
    for parameter in q.parameters():
        parameter.data.fill_(float("nan"))
    q.prefix_encoder.rotary_emb.inv_freq.fill_(float("nan"))
    q.init_weights()
    assert all(torch.isfinite(parameter).all() for parameter in q.parameters())
    rope = q.prefix_encoder.rotary_emb
    expected_freq, expected_scale = rope.rope_init_fn(rope.config, rope.inv_freq.device)
    torch.testing.assert_close(rope.inv_freq, expected_freq)
    torch.testing.assert_close(rope.original_inv_freq, expected_freq)
    assert rope.attention_scaling == expected_scale
    assert all(parameter.requires_grad for parameter in q.parameters())
    scores = q(torch.tensor([[4, 5, 6, 7]]), attention_mask=torch.zeros(1, 1, 4, 4))
    assert scores.shape == (1, 4)
    scores.square().sum().backward()
    assert q.q_head.score.weight.grad.norm() > 0
    assert q.prefix_encoder.embed_tokens.weight.grad.norm() > 0


def test_q_config_only_creation_and_export_are_self_contained(tmp_path):
    path = q_config_file(tmp_path)
    # There is deliberately no model.safetensors or accessible teacher checkpoint.
    q = build_q_model(path, init_device="cpu", attn_implementation="eager")
    assert not (tmp_path / "model.safetensors").exists()
    inputs = torch.tensor([[4, 5, 6, 7]])
    mask = torch.zeros(1, 1, 4, 4)
    scores = q(inputs, attention_mask=mask)
    package = tmp_path / "exported"
    save_q_weights(q, q.state_dict(), package)
    restored = load_q_model(package, attn_implementation="eager")
    torch.testing.assert_close(scores, restored(inputs, attention_mask=mask), atol=0, rtol=0)
    assert set(q.state_dict()) == set(restored.state_dict())
    main = Qwen3ForBDLM(tiny_config())
    prepare_p_head(main)
    save_recipe_assets(main, q, tmp_path / "assets")
    assert (tmp_path / "assets" / "p_head_config.json").is_file()
    assert (tmp_path / "assets" / "q_model" / "q_config.json").is_file()


def test_q_checkpointing_preserves_outputs_and_gradients(tmp_path):
    path = q_config_file(tmp_path)
    plain = build_q_model(path, init_device="cpu", attn_implementation="eager")
    checked = build_q_model(path, init_device="cpu", attn_implementation="eager")
    checked.load_state_dict(plain.state_dict())
    checked.gradient_checkpointing_enable({"use_reentrant": False})
    inputs = torch.tensor([[4, 5, 6, 7]])
    mask = torch.zeros(1, 1, 4, 4)
    a = plain(inputs, attention_mask=mask)
    b = checked(inputs, attention_mask=mask)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.square().sum().backward()
    b.square().sum().backward()
    for (name, left), (_, right) in zip(plain.named_parameters(), checked.named_parameters()):
        assert left.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


def test_q_architecture_rejects_missing_or_guessed_configuration(tmp_path):
    path = q_config_file(tmp_path, base_config={}, encoder_num_layers=1)
    with pytest.raises(ValueError, match="missing architecture"):
        load_q_config(path)
    path = q_config_file(tmp_path, encoder_num_layers=10)
    with pytest.raises(ValueError, match="exceeds"):
        load_q_config(path)


def test_veomni_cpu_weight_loader_initializes_missing_meta_head(tmp_path):
    from veomni.models.module_utils import load_model_weights

    torch.manual_seed(31)
    original = Qwen3ForBDLM(tiny_config()).eval()
    provider = Qwen3Provider()
    provider.save_weights(original, Tokenizer(), original.state_dict(), tmp_path)
    # Bare Qwen3 exports retain the generic provider's loader name.
    assert AutoConfig.from_pretrained(tmp_path).auto_map["AutoModelForCausalLM"] == (
        "modeling_dfactory_qwen3.Qwen3ForBDLM"
    )
    assert (tmp_path / "modeling_dfactory_qwen3.py").is_file()
    restored = provider.build_model(ModelConfig(config_path=str(tmp_path), attention="eager"))
    prepare_p_head(restored)
    assert restored.model.embed_tokens.weight.is_meta
    assert not restored.model.rotary_emb.inv_freq.is_meta
    assert next(restored.p_head.parameters()).is_meta

    # This is the real pinned VeOmni 0.1.4 missing-key loader, on CPU.
    load_model_weights(restored, str(tmp_path), init_device="cpu")
    restored.eval()
    assert all(torch.isfinite(parameter).all() for parameter in restored.p_head.parameters())
    torch.testing.assert_close(restored.p_head.norm.weight, torch.ones_like(restored.p_head.norm.weight))
    assert abs(float(restored.p_head.gate_proj.weight.std()) - restored.p_head.initializer_std) < 0.005
    ids = torch.tensor([[4, 5, 6, 7]])
    mask = torch.zeros(1, 1, 4, 4)
    torch.testing.assert_close(
        original(ids, attention_mask=mask, use_cache=False).logits,
        restored(ids, attention_mask=mask, use_cache=False).logits,
        atol=0,
        rtol=0,
    )


def test_q_inline_config_preserves_metadata_without_remote_or_teacher_dependencies(tmp_path):
    path = q_config_file(
        tmp_path,
        format="mask_generator_q_linear.init.v1",
        teacher_config_path="/not/a/real/teacher/config.json",
        teacher_model_path="/not/a/real/teacher/weights",
        auto_map={"AutoModel": "unavailable_remote_code.MaskGeneratorQLinearForOrdering"},
        eos_token_id=151645,
        attention_block_size=32,
        attention_mask_type="doc_causal_block",
        init_source="teacher_prefix_plus_linear_head",
    )
    config = load_q_config(path)
    assert config["attention_block_size"] == 32
    assert config["eos_token_id"] == 151645
    assert config["teacher_model_path"] == "/not/a/real/teacher/weights"
    assert "auto_map" not in config
    assert config["source_auto_map"]["AutoModel"].startswith("unavailable_remote_code.")
    model = build_q_model(path, init_device="meta", attn_implementation="eager")
    assert next(model.parameters()).is_meta
    assert len(model.prefix_encoder.layers) == 1


@pytest.mark.parametrize(("profile", "layers", "hidden_size"), [("1.7b", 14, 2048), ("8b", 12, 4096)])
def test_bundled_qwen3_q_architecture_builds_configured_layers_on_meta(profile, layers, hidden_size):
    path = Path(__file__).parents[2] / "configs" / "q_configs" / f"qwen3_{profile}_soldiff.json"
    config = load_q_config(path)
    assert config["encoder_num_layers"] == layers
    assert config["base_config"]["pad_token_id"] is None
    model = build_q_model(path, init_device="meta", attn_implementation="flex_attention")
    assert len(model.prefix_encoder.layers) == layers
    assert model.prefix_encoder.embed_tokens.weight.shape == (151936, hidden_size)
    assert model.q_head.score.weight.shape == (1, hidden_size)
    assert model.q_head.score.bias is None
    assert next(model.parameters()).is_meta


@pytest.mark.parametrize(
    ("profile", "layers", "hidden_size", "p_intermediate", "tied"),
    [("1.7b", 28, 2048, 1024, True), ("8b", 36, 4096, 2048, False)],
)
def test_bundled_main_profiles_build_matching_p_heads_on_meta(
    profile, layers, hidden_size, p_intermediate, tied, monkeypatch
):
    root = Path(__file__).parents[2]
    monkeypatch.chdir(root)
    run = load_config(root / "configs" / "recipes" / "soldiff" / f"qwen3_{profile}.yaml")
    model = Qwen3Provider().build_model(run.model)
    resolved = prepare_p_head(model, p_head_config_path=run.recipe.params["p_head_config_path"])
    assert resolved == json.loads(Path(run.recipe.params["p_head_config_path"]).read_text())
    assert len(model.model.layers) == layers
    assert model.model.embed_tokens.weight.shape == (151936, hidden_size)
    assert model.lm_head.weight.shape == (151936, hidden_size)
    assert model.config.tie_word_embeddings is tied
    if not tied:
        assert model.model.embed_tokens.weight is not model.lm_head.weight
    assert model.p_head.gate_proj.weight.shape == (p_intermediate, hidden_size)
    assert model.p_head.down_proj.weight.shape == (1, p_intermediate)
    assert all(parameter.is_meta for parameter in model.parameters())


@pytest.mark.parametrize(("rank", "prefix"), [(0, "p_head."), (1, "model.p_head.")])
def test_separate_p_head_weights_preserve_local_fsdp2_shards(tmp_path, rank, prefix):
    # Each rank uses an isolated interpreter because DeviceMesh caches groups.
    # The fake process group provides topology only. distribute_tensor with
    # src_data_rank=None performs local slicing, so no fake collective result
    # participates in the tensor equality assertions below.
    code = dedent(
        """
        import sys
        from pathlib import Path

        import torch
        import torch.distributed as dist
        from safetensors.torch import save_file
        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.device_mesh import init_device_mesh
        from torch.testing._internal.distributed.fake_pg import FakeStore

        from dfactory.models.qwen3.heads import TokenScoreMLP
        from dfactory.recipes.soldiff.artifacts import load_p_head_weights

        target, rank, prefix = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
        dist.init_process_group('fake', store=FakeStore(), rank=rank, world_size=2)
        try:
            model = torch.nn.Module()
            model.p_head = TokenScoreMLP(hidden_size=4, intermediate_size=3)
            expected = {
                name: torch.arange(value.numel(), dtype=value.dtype).reshape(value.shape) + 0.25
                for name, value in model.p_head.state_dict().items()
            }
            save_file({prefix + name: value for name, value in expected.items()}, target / 'p_head.safetensors')
            mesh = init_device_mesh('cpu', (2,))
            fully_shard(model.p_head, mesh=mesh)
            parameter_ids = {name: id(value) for name, value in model.p_head.named_parameters()}
            load_p_head_weights(model, model_path=str(target))
            for name, actual in model.p_head.state_dict().items():
                # The scalar scorer's [1, 3] weight also covers an empty shard
                # on rank 1, and the [3, 4] projections cover uneven sharding.
                shard = torch.tensor_split(expected[name], 2, dim=0)[rank]
                torch.testing.assert_close(actual.to_local(), shard, rtol=0, atol=0)
            assert parameter_ids == {name: id(value) for name, value in model.p_head.named_parameters()}
        finally:
            dist.destroy_process_group()
        """
    )
    env = dict(os.environ)
    root = Path(__file__).parents[2]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root / "src"), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(rank), prefix],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
