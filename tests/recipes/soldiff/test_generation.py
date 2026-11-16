import ast
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import PreTrainedTokenizerFast, Qwen3Config

from dfactory.models.qwen3 import Qwen3ForBDLM, Qwen3Provider
from dfactory.recipes.soldiff.generation import (
    SolDiffGenerationConfig,
    _attention_mask,
    _fork_cache,
    _predict_tokens,
    _select_positions,
    generate_soldiff,
    load_soldiff_model,
)


def generation_config(**kwargs):
    return replace(
        SolDiffGenerationConfig(
            max_new_tokens=7, block_length=4, steps_per_block=4, mask_token_id=31, eos_token_id=None
        ),
        **kwargs,
    )


def tiny_model(attention="eager"):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=None,
        eos_token_id=2,
        p_head_config=dict(
            hidden_size=16,
            intermediate_size=7,
            activation="gelu",
            rms_eps=3e-6,
            initializer_std=0.017,
        ),
    )
    config._attn_implementation = attention
    torch.manual_seed(137)
    return Qwen3ForBDLM(config).eval()


@pytest.mark.parametrize(
    "values",
    [
        {"strategy": "other"},
        {"block_length": 0},
        {"steps_per_block": -1},
        {"max_new_tokens": -1},
        {"prior_temperature": -1},
        {"temperature": float("nan")},
        {"confidence_threshold": 1.1},
        {"top_k": -1},
        {"top_p": 0},
        {"eos_token_id": [True]},
        {"mask_token_id": 2, "eos_token_id": 2},
    ],
)
def test_invalid_generation_settings(values):
    with pytest.raises(ValueError):
        generation_config(**values).validate()


def test_generation_defaults_keep_dual_eos_separate_from_model_config():
    assert SolDiffGenerationConfig().eos_ids == (151643, 151645)
    assert SolDiffGenerationConfig(eos_token_id=[2, 3, 2]).eos_ids == (2, 3)
    assert not SolDiffGenerationConfig().eos_early_stop


@pytest.mark.parametrize("temperature,top_k,top_p", [(0, None, None), (0.5, 3, 0.8), (2, 1, 1)])
def test_token_choice_is_argmax_with_float32_confidence(temperature, top_k, top_p):
    logits = torch.tensor([[[0.1, 1.1, 0.7, -0.3]]], dtype=torch.bfloat16)
    config = generation_config(temperature=temperature, top_k=top_k, top_p=top_p)
    tokens, confidence = _predict_tokens(logits, config)
    expected = logits.float() / (temperature or 1)
    if top_k:
        expected[expected < expected.topk(top_k).values[..., -1, None]] = -torch.inf
    if top_p and top_p < 1:
        values, indices = expected.sort(descending=True)
        remove = values.softmax(-1).cumsum(-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        expected.masked_fill_(torch.zeros_like(remove).scatter(-1, indices, remove), -torch.inf)
    assert tokens.tolist() == [[1]]
    assert confidence.dtype == torch.float32
    torch.testing.assert_close(confidence, expected.softmax(-1)[..., 1])


@pytest.mark.parametrize("floor,expected", [(0, [0]), (1, [0]), (2, [0, 1])])
def test_dynamic_accepts_only_contiguous_high_confidence_prefix(floor, expected):
    selected = _select_positions(
        torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
        torch.tensor([[0.95, 0.9, 0.99, 1.0]]),
        torch.tensor([[True, True, True, False]]),
        floor,
        generation_config(strategy="prior_confidence_dynamic"),
        None,
    )
    assert selected.nonzero(as_tuple=True)[1].tolist() == expected


@pytest.mark.parametrize("strategy", ["prior_static", "prior_confidence_dynamic"])
@pytest.mark.parametrize("prior_temperature", [1.0, 1e-20])
def test_position_sampling_never_selects_inactive_slots(strategy, prior_temperature):
    config = generation_config(strategy=strategy, prior_temperature=prior_temperature, confidence_threshold=1)
    scores = torch.tensor([[0.0, 4.0, 0.0, -2.0]])
    active = torch.tensor([[False, True, False, True]])
    selected = _select_positions(scores, torch.ones_like(scores), active, 2, config, torch.Generator().manual_seed(5))
    assert torch.equal(selected, active)


@pytest.mark.parametrize("strategy", ["prior_static", "prior_confidence_dynamic"])
def test_shared_prior_temperature_controls_position_sampling(strategy):
    scores = torch.tensor([[0.0, 0.1, 0.2, 0.3]])
    config = generation_config(strategy=strategy, prior_temperature=1, confidence_threshold=1)
    choices = {
        int(
            _select_positions(
                scores,
                torch.ones_like(scores),
                torch.ones_like(scores, dtype=torch.bool),
                1,
                config,
                torch.Generator().manual_seed(seed),
            ).nonzero()[0, 1]
        )
        for seed in range(10)
    }
    assert len(choices) > 1


class ToyCache:
    def __init__(self):
        self.layers = [SimpleNamespace(keys=torch.empty(1, 0, dtype=torch.long))]

    def get_seq_length(self):
        return self.layers[0].keys.shape[1]


class PositionHead(torch.nn.Module):
    def forward(self, hidden):
        return hidden[..., 0]


class ToyModel(torch.nn.Module):
    def __init__(self, predictions=None, scores=None):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.p_head = PositionHead()
        self.predictions = predictions or {}
        self.scores = scores or {}
        self.calls = []

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        cache_position=None,
        past_key_values=None,
        use_cache=False,
        output_hidden_states=False,
        **kwargs,
    ):
        prefix_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        positions = position_ids[0].tolist()
        expected = torch.tensor(
            [[key // 4 <= query // 4 for key in range(prefix_length + len(positions))] for query in positions]
        )
        assert torch.equal(attention_mask[0, 0] == 0, expected)
        if cache_position is not None:
            assert torch.equal(cache_position, position_ids[0])
        self.calls.append((input_ids.clone(), position_ids.clone(), use_cache, output_hidden_states))
        hidden = torch.tensor([[[self.scores.get(pos, -float(pos))] for pos in positions]])
        logits = torch.full((1, len(positions), 32), -10.0)
        for index, pos in enumerate(positions):
            logits[0, index, self.predictions.get(pos, 5 + pos % 10)] = 10.0
        if use_cache:
            past_key_values = past_key_values or ToyCache()
            past_key_values.layers[0].keys = torch.cat((past_key_values.layers[0].keys, input_ids), dim=1)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,), past_key_values=past_key_values)


@pytest.mark.parametrize("prompt_length", [0, 1, 3, 4, 5, 8])
@pytest.mark.parametrize("use_cache", [True, False])
def test_block_boundaries_absolute_positions_and_final_full_mask_block(prompt_length, use_cache):
    model = ToyModel()
    prompt = torch.full((1, prompt_length), 3)
    result = generate_soldiff(model, prompt, generation_config(use_cache=use_cache))
    assert result.tolist() == [[3] * prompt_length + [5 + pos % 10 for pos in range(prompt_length, prompt_length + 7)]]
    denoising = [call for call in model.calls if call[3]]
    total = ((prompt_length + 7 + 3) // 4) * 4
    last = [call for call in denoising if call[1][0, -1] == total - 1]
    assert last
    tail = prompt_length + 7
    for ids, positions, _, _ in last:
        assert not (ids[positions >= tail] == 0).any()
    assert last[-1][0][0, -1] == 31  # Tail slots participate in the denoising schedule.


@pytest.mark.parametrize("eos", [2, 3])
@pytest.mark.parametrize("use_cache", [True, False])
def test_early_stop_waits_for_complete_prefix_and_rechecks_previous_eos(eos, use_cache):
    model = ToyModel(predictions={2: eos}, scores={2: 4, 1: 3, 0: 2, 3: 1})
    prompt = torch.tensor([[2]])  # An EOS in the prompt must not stop generation.
    result = generate_soldiff(
        model,
        prompt,
        generation_config(
            use_cache=use_cache,
            eos_token_id=[2, 3],
            eos_early_stop=True,
        ),
    )
    assert result.tolist() == [[2, 6, eos]]
    assert len([call for call in model.calls if call[3]]) == 2


def test_block_end_eos_and_tail_eos_behavior():
    model = ToyModel(predictions={2: 2})
    result = generate_soldiff(model, torch.tensor([[3]]), generation_config(eos_token_id=2))
    assert result.tolist() == [[3, 6, 2]]
    assert len([call for call in model.calls if call[3]]) == 3
    tail_model = ToyModel(predictions={3: 2})
    result = generate_soldiff(tail_model, torch.tensor([[3]]), generation_config(max_new_tokens=1, eos_token_id=2))
    assert result.tolist() == [[3, 6]]
    assert tail_model.calls[0][0].tolist() == [[3, 31, 31, 31]]


def test_empty_generation_and_input_contract():
    prompt = torch.tensor([[3]])
    assert torch.equal(generate_soldiff(ToyModel(), prompt, generation_config(max_new_tokens=0)), prompt)
    for invalid in (torch.tensor([3]), torch.tensor([[31]]), torch.tensor([[3], [4]]), torch.tensor([[3.0]])):
        with pytest.raises(ValueError):
            generate_soldiff(ToyModel(), invalid, generation_config())


@pytest.mark.parametrize("strategy", ["prior_static", "prior_confidence_dynamic"])
@pytest.mark.parametrize("prompt_length", [1, 4, 5])
def test_real_qwen_cache_matches_full_prefix_with_sdpa_and_eager(strategy, prompt_length):
    outputs = []
    for attention in ("eager", "sdpa"):
        model = tiny_model(attention)
        model.config.return_dict = False
        for cache in (False, True):
            outputs.append(
                generate_soldiff(
                    model,
                    torch.tensor([[3] * prompt_length]),
                    generation_config(
                        strategy=strategy,
                        use_cache=cache,
                        top_k=1,
                    ),
                )
            )
    for output in outputs[1:]:
        assert torch.equal(output, outputs[0])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("strategy", ["prior_static", "prior_confidence_dynamic"])
def test_low_precision_sdpa_generation_preserves_mask_and_cache(dtype, strategy, monkeypatch):
    sdpa = torch.nn.functional.scaled_dot_product_attention
    shapes = []

    def check_sdpa(query, key, value, *, attn_mask, **kwargs):
        if attn_mask.dtype != query.dtype:
            raise RuntimeError("invalid dtype for bias - should match query's dtype")
        query_length, key_length = query.shape[-2], key.shape[-2]
        queries = torch.arange(key_length - query_length, key_length) // 4
        keys = torch.arange(key_length) // 4
        allowed = keys[None, :] <= queries[:, None]
        assert torch.equal(attn_mask[0, 0] == 0, allowed)
        assert torch.isneginf(attn_mask[0, 0][~allowed]).all()
        shapes.append((query_length, key_length))
        return sdpa(query, key, value, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", check_sdpa)
    model = tiny_model("sdpa").to(dtype=dtype)
    prompt = torch.tensor([[3, 4, 5, 6, 7]])
    config = generation_config(strategy=strategy, top_k=1)
    cached = generate_soldiff(model, prompt, config)
    full = generate_soldiff(model, prompt, replace(config, use_cache=False))
    assert torch.equal(cached, full)
    assert (4, 4) in shapes  # Prompt prefill.
    assert (4, 8) in shapes  # Cached block decoding and cache updates.
    assert (8, 8) in shapes  # Decoding without a prefix cache.


def test_real_cache_fork_preserves_prefix_tensors():
    model = tiny_model()
    inputs = torch.tensor([[3, 4, 5, 6]])
    mask = _attention_mask(model, 0, 4, 4, 4, inputs.device)
    cache = model(input_ids=inputs, attention_mask=mask, use_cache=True).past_key_values
    keys = cache.layers[0].keys.clone()
    fork = _fork_cache(cache)
    model(
        input_ids=inputs,
        attention_mask=_attention_mask(model, 4, 4, 8, 4, inputs.device),
        position_ids=torch.arange(4, 8)[None],
        past_key_values=fork,
        use_cache=True,
    )
    assert cache.get_seq_length() == 4
    assert fork.get_seq_length() == 8
    torch.testing.assert_close(cache.layers[0].keys, keys, atol=0, rtol=0)


def test_flex_mask_uses_absolute_query_positions_and_guards_padding():
    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="flex_attention"))
    mask = _attention_mask(model, 4, 4, 8, 4, torch.device("cpu"))
    for query, key, expected in [(0, 0, True), (0, 7, True), (3, 7, True), (4, 0, False), (0, 8, False)]:
        actual = mask.mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor(query), torch.tensor(key))
        assert bool(actual) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("attention", ["sdpa", "flex_attention"])
def test_cuda_generation_cache_matches_full_prefix(attention):
    model = tiny_model(attention).to(device="cuda", dtype=torch.bfloat16)
    prompt = torch.tensor([[3, 4, 5, 6, 7]], device="cuda")
    config = generation_config(strategy="prior_confidence_dynamic", top_k=1)
    cached = generate_soldiff(model, prompt, config)
    full = generate_soldiff(model, prompt, replace(config, use_cache=False))
    assert torch.equal(cached, full)


@pytest.mark.parametrize(
    "sidecar,sharded,prefix", [(False, False, False), (True, False, False), (False, True, False), (True, False, True)]
)
def test_loader_restores_head_architecture_and_exact_weights(tmp_path, sidecar, sharded, prefix):
    model = tiny_model()
    with torch.no_grad():
        for index, parameter in enumerate(model.p_head.parameters()):
            parameter.fill_(0.125 * (index + 1))
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    model.save_pretrained(tmp_path, max_shard_size="4KB" if sharded else "5GB")
    if sidecar:
        config = json.loads((tmp_path / "config.json").read_text())
        (tmp_path / "p_head_config.json").write_text(json.dumps(config.pop("p_head_config")))
        (tmp_path / "config.json").write_text(json.dumps(config))
    if prefix:
        weights = load_file(tmp_path / "model.safetensors")
        save_file(
            {("model." + key if key.startswith("p_head.") else key): value for key, value in weights.items()},
            tmp_path / "model.safetensors",
        )
    restored = load_soldiff_model(tmp_path, device="cpu", torch_dtype=torch.float32, attn_implementation="eager")
    assert restored.p_head.intermediate_size == 7
    assert restored.p_head.activation == "gelu"
    assert restored.p_head.rms_eps == 3e-6
    for name, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)


def test_loader_rejects_absent_head_weights_and_conflicting_configs(tmp_path):
    model = tiny_model()
    model.save_pretrained(tmp_path)
    sidecar = tmp_path / "p_head_config.json"
    sidecar.write_text(json.dumps({**model.config.p_head_config, "activation": "silu"}))
    with pytest.raises(ValueError, match="conflicts"):
        load_soldiff_model(tmp_path, device="cpu")
    sidecar.unlink()
    weights_path = tmp_path / "model.safetensors"
    weights = load_file(weights_path)
    save_file({name: value for name, value in weights.items() if not name.startswith("p_head.")}, weights_path)
    with pytest.raises(ValueError, match="p_head"):
        load_soldiff_model(tmp_path, device="cpu", torch_dtype=torch.float32)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("p_head_config")
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="p_head_config"):
        load_soldiff_model(tmp_path, device="cpu")


def test_separate_head_weights_do_not_replace_complete_main_head(tmp_path):
    model = tiny_model()
    model.save_pretrained(tmp_path)
    save_file(
        {name: torch.zeros_like(value) for name, value in model.p_head.state_dict().items()},
        tmp_path / "p_head.safetensors",
    )
    restored = load_soldiff_model(tmp_path, device="cpu", torch_dtype=torch.float32)
    for name, value in model.p_head.state_dict().items():
        torch.testing.assert_close(restored.p_head.state_dict()[name], value, atol=0, rtol=0)


def test_loader_can_restore_complete_separate_head_weights(tmp_path):
    model = tiny_model()
    model.save_pretrained(tmp_path)
    weights_path = tmp_path / "model.safetensors"
    weights = load_file(weights_path)
    save_file({name: value for name, value in weights.items() if not name.startswith("p_head.")}, weights_path)
    save_file(dict(model.p_head.state_dict()), tmp_path / "p_head.safetensors")
    restored = load_soldiff_model(tmp_path, device="cpu", torch_dtype=torch.float32)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)


def test_tied_embeddings_load_without_losing_p_weights(tmp_path):
    model = tiny_model()
    model.config.tie_word_embeddings = True
    model.tie_weights()
    model.save_pretrained(tmp_path)
    restored = load_soldiff_model(tmp_path, device="cpu", torch_dtype=torch.float32)
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    for name, value in model.p_head.state_dict().items():
        torch.testing.assert_close(restored.p_head.state_dict()[name], value, atol=0, rtol=0)


def test_export_preserves_model_math_and_decoding_source(tmp_path):
    from dfactory.recipes.soldiff.export import export_soldiff_code

    root = Path(__file__).resolve().parents[3] / "src" / "dfactory"
    recipe = root / "recipes" / "soldiff"
    model = root / "models" / "qwen3"
    for filename in ("heads_soldiff.py", "hf_soldiff.py"):
        (tmp_path / filename).write_text("unused\n")
    export_soldiff_code(tmp_path, tiny_model().config.to_dict())
    assert {path.name for path in tmp_path.glob("*.py")} == {"modeling_soldiff.py", "generation_soldiff.py"}
    for source, destination, excluded in (
        (model / "modeling.py", tmp_path / "modeling_soldiff.py", {"Qwen3Model", "Qwen3ForBDLM"}),
        (recipe / "generation.py", tmp_path / "generation_soldiff.py", {"_head_config", "load_soldiff_model"}),
        (model / "heads.py", tmp_path / "modeling_soldiff.py", {"get_p_head"}),
        (model / "heads.py", tmp_path / "generation_soldiff.py", {
            "qwen_style_rms_norm", "QwenStyleRMSNorm", "TokenScoreMLP", "PWarmupHeadConfig",
            "_first_parameter_device", "_get_decoder", "build_p_head_config", "attach_p_head",
        }),
        (recipe / "hf.py", tmp_path / "modeling_soldiff.py", {"SolDiffGenerationMixin"}),
        (recipe / "hf.py", tmp_path / "generation_soldiff.py", {"causallm_loss_function"}),
    ):
        original = ast.parse(source.read_text())
        exported = ast.parse(destination.read_text())
        definitions = {node.name: node for node in exported.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for node in original.body:
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) or node.name in excluded:
                continue
            if node.name == "generate_soldiff":
                node.body = [
                    child for child in node.body
                    if not (isinstance(child, ast.ImportFrom) and child.module == "dfactory.models.qwen3.heads")
                ]
            assert ast.dump(node) == ast.dump(definitions[node.name]), node.name
    for path in tmp_path.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module.split(".")[0] not in {"dfactory", "veomni", "liger_kernel"}
            elif isinstance(node, ast.Import):
                assert not {alias.name.split(".")[0] for alias in node.names} & {"dfactory", "veomni", "liger_kernel"}


@pytest.mark.parametrize("attention,dtype", [("eager", torch.float32), ("sdpa", torch.bfloat16)])
def test_standalone_hf_export_matches_forward_and_generation(tmp_path, attention, dtype):
    from dataclasses import asdict

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    model = tiny_model(attention).to(dtype=dtype)
    model.config.mask_token_id = 31
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", eos_token="t2", mask_token="t31")
    tokenizer.chat_template = "{{ messages[0]['content'] }}{% if add_generation_prompt %} t4{% endif %}"
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint, max_shard_size="4KB")
    Qwen3Provider().save_assets(model, tokenizer, checkpoint)
    model = load_soldiff_model(checkpoint, device="cpu", torch_dtype=dtype, attn_implementation=attention)
    prompt = torch.tensor([[3, 4, 5, 6, 7]])
    with torch.inference_mode():
        output = model(
            input_ids=prompt,
            attention_mask=_attention_mask(model, 0, 5, 5, 4, prompt.device),
            output_hidden_states=True,
            use_cache=False,
        )
        expected = {
            "state": model.state_dict(),
            "logits": output.logits,
            "scores": model.p_head(output.hidden_states[-1]),
            "cases": [],
        }
        for strategy in ("prior_static", "prior_confidence_dynamic"):
            for cache in (False, True):
                for temperature in (0, 0.7):
                    config = generation_config(strategy=strategy, use_cache=cache, prior_temperature=temperature)
                    tokens = generate_soldiff(model, prompt, config, generator=torch.Generator().manual_seed(17))
                    expected["cases"].append((asdict(config), tokens))
        first_token = expected["cases"][0][1][0, prompt.shape[1]].item()
        for early_stop in (False, True):
            config = generation_config(eos_token_id=[3, first_token], eos_early_stop=early_stop)
            expected["cases"].append((asdict(config), generate_soldiff(model, prompt, config)))
        config = generation_config(max_new_tokens=0)
        expected["cases"].append((asdict(config), prompt))
    reference = tmp_path / "expected.pt"
    torch.save(expected, reference)
    code = dedent(
        """
        import importlib.abc
        import importlib.util
        import os
        import sys

        if os.environ.get('SOLDIFF_STANDALONE_PYTHON'):
            assert importlib.util.find_spec('dfactory') is None
            assert importlib.util.find_spec('veomni') is None

        class InferenceOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'dfactory', 'veomni'}:
                    raise ModuleNotFoundError(f'Training dependency requested: {fullname}')

        sys.meta_path.insert(0, InferenceOnly())
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        path, reference, attention, dtype = sys.argv[1:]
        expected = torch.load(reference, weights_only=True)
        model, info = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, local_files_only=True,
            attn_implementation=attention, dtype=getattr(torch, dtype), output_loading_info=True,
        )
        model.eval()
        assert model.generate.__module__.endswith('.generation_soldiff')
        assert not info['missing_keys'] and not info['unexpected_keys']
        for name, tensor in expected['state'].items():
            torch.testing.assert_close(model.state_dict()[name], tensor, atol=0, rtol=0)
        prompt = torch.tensor([[3, 4, 5, 6, 7]])
        allowed = torch.arange(5)[None, :] // 4 <= torch.arange(5)[:, None] // 4
        mask = torch.zeros(1, 1, 5, 5, dtype=model.dtype).masked_fill(~allowed, -torch.inf)
        with torch.inference_mode():
            output = model(prompt, attention_mask=mask, output_hidden_states=True, use_cache=False)
            torch.testing.assert_close(output.logits, expected['logits'], atol=0, rtol=0)
            torch.testing.assert_close(model.p_head(output.hidden_states[-1]), expected['scores'], atol=0, rtol=0)
        for config, tokens in expected['cases']:
            actual = model.generate(
                input_ids=prompt, attention_mask=torch.ones_like(prompt),
                generator=torch.Generator().manual_seed(17), **config,
            )
            assert torch.equal(actual, tokens)
            actual = model.generate(prompt, generation_config=config, generator=torch.Generator().manual_seed(17))
            assert torch.equal(actual, tokens)
        native_config, tokens = expected['cases'][0]
        model.config.soldiff_generation_config = dict(native_config)
        assert torch.equal(model.generate(prompt), tokens)
        hf_config = GenerationConfig(**{**native_config, 'temperature': 1, 'do_sample': False})
        assert torch.equal(model.generate(prompt, generation_config=hf_config), tokens)
        for kwargs in (
            {'num_beams': 2}, {'attention_mask': torch.zeros_like(prompt)},
            {'generation_config': GenerationConfig(repetition_penalty=1.2)},
        ):
            try:
                model.generate(prompt, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError('Unsupported generation settings were silently accepted')
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        assert tokenizer('t3 t4')['input_ids'] == [3, 4]
        chat = tokenizer.apply_chat_template([{'role': 'user', 'content': 't3'}], tokenize=False,
                                             add_generation_prompt=True)
        assert chat == 't3 t4'
        decoder = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True,
                                            attn_implementation=attention, dtype=getattr(torch, dtype))
        assert type(decoder).__name__ == 'Qwen3Model'
        copied_path = path + '_copy'
        model.save_pretrained(copied_path, max_shard_size='4KB')
        tokenizer.save_pretrained(copied_path)
        reloaded = AutoModelForCausalLM.from_pretrained(copied_path, trust_remote_code=True, local_files_only=True,
                                                       attn_implementation=attention, dtype=getattr(torch, dtype))
        assert torch.equal(reloaded.generate(prompt), tokens)
        assert not any(name.split('.')[0] in {'dfactory', 'veomni', 'liger_kernel'} for name in sys.modules)
        print('Standalone weights, forward, p scores, tokenizer, and both decoding strategies passed')
        """
    )
    environment = dict(os.environ, HF_HOME=str(tmp_path / "hf_cache"), HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1")
    result = subprocess.run(
        [
            os.environ.get("SOLDIFF_STANDALONE_PYTHON", sys.executable),
            "-I",
            "-c",
            code,
            str(checkpoint),
            str(reference),
            attention,
            str(dtype).removeprefix("torch."),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def generation_cli():
    script = Path(__file__).resolve().parents[3] / "scripts" / "soldiff" / "generate.py"
    spec = importlib.util.spec_from_file_location("generate_soldiff_script", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


@pytest.mark.parametrize("flag", ["--enable-thinking", "--no-enable-thinking"])
def test_thinking_options_require_chat_template(generation_cli, flag, capsys):
    with pytest.raises(SystemExit) as error:
        generation_cli.parse_args(["--model-path", "checkpoint", "--prompt", "Hello", flag])
    assert error.value.code == 2
    assert "requires --chat-template" in capsys.readouterr().err


@pytest.mark.parametrize("strategy", ["prior_static", "prior_confidence_dynamic"])
@pytest.mark.parametrize("chat_template,thinking", [(False, None), (True, None), (True, True), (True, False)])
def test_cli_generates_and_saves_shared_settings(
    tmp_path, capsys, monkeypatch, generation_cli, strategy, chat_template, thinking
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    cli = generation_cli
    model = tiny_model()
    with torch.no_grad():
        model.model.embed_tokens.weight.fill_(0.1)
        model.lm_head.weight.zero_()
        model.lm_head.weight[5].fill_(1.0)
    model.save_pretrained(tmp_path)
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", eos_token="t2", mask_token="t31")
    fast.chat_template = (
        "{% if enable_thinking is not defined %}t9 "
        "{% elif enable_thinking %}t10 {% else %}t11 {% endif %}"
        "{{ messages[0]['content'] }}{% if add_generation_prompt %} t12{% endif %}"
    )
    fast.save_pretrained(tmp_path)
    template_calls = []
    apply_chat_template = PreTrainedTokenizerFast.apply_chat_template

    def capture_chat_template(tokenizer, conversation, **kwargs):
        template_calls.append((conversation, dict(kwargs)))
        return apply_chat_template(tokenizer, conversation, **kwargs)

    monkeypatch.setattr(PreTrainedTokenizerFast, "apply_chat_template", capture_chat_template)
    generation_inputs = []
    generate = cli.generate_soldiff

    def capture_generate(model, input_ids, config):
        generation_inputs.append(input_ids.clone())
        return generate(model, input_ids, config)

    monkeypatch.setattr(cli, "generate_soldiff", capture_generate)
    output = tmp_path / "result.json"
    chat_args = ["--chat-template"] if chat_template else []
    if thinking is not None:
        chat_args.append("--enable-thinking" if thinking else "--no-enable-thinking")
    cli.main(
        [
            "--model-path",
            str(tmp_path),
            "--prompt",
            "t3 t4 t6 t7 t8",
            "--output",
            str(output),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--attention",
            "eager",
            "--mask-token-id",
            "31",
            "--eos-token-id",
            "2",
            "3",
            "--max-new-tokens",
            "5",
            "--block-length",
            "4",
            "--steps-per-block",
            "4",
            "--strategy",
            strategy,
            "--prior-temperature",
            "0.5",
            *chat_args,
        ]
    )
    record = json.loads(output.read_text())
    assert record["text"] == "t5 t5 t5 t5 t5"
    assert record["completion_ids"] == [5] * 5
    assert record["generation_config"]["prior_temperature"] == 0.5
    assert record["generation_config"]["strategy"] == strategy
    assert record["generation_config"]["eos_token_id"] == [2, 3]
    assert record["chat_template"] is chat_template
    assert record["enable_thinking"] is thinking
    prompt_ids = [3, 4, 6, 7, 8]
    if chat_template:
        assert len(template_calls) == 1
        conversation, kwargs = template_calls[0]
        assert conversation == [{"role": "user", "content": "t3 t4 t6 t7 t8"}]
        assert kwargs["add_generation_prompt"] is True
        if thinking is None:
            assert "enable_thinking" not in kwargs
        else:
            assert kwargs["enable_thinking"] is thinking
        prompt_ids = [{None: 9, True: 10, False: 11}[thinking], *prompt_ids, 12]
    else:
        assert not template_calls
    assert generation_inputs[0].tolist() == [prompt_ids]
    assert "t5 t5 t5 t5 t5" in capsys.readouterr().out
    assert cli.decode_completion(fast, torch.tensor([5, 6, 7, 2]), ["t7", "t6"]) == "t5"
