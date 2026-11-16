"""SolDiff training updates, checkpoint recovery and public metrics."""

import json
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config

from dfactory.config import (
    CheckpointConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
    RecipeConfig,
    RunConfig,
    RunSettings,
)
from dfactory.models.qwen3 import Qwen3ForBDLM
from dfactory.recipes import get_recipe
from dfactory.recipes.soldiff.config import SolDiffConfig
from dfactory.recipes.soldiff.trainer import SolDiffTrainer
from dfactory.runtime import RuntimeContext
from dfactory.runtime.veomni import VeOmniGateway


class TinyMain(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 4)
        self.lm_head = torch.nn.Linear(4, 13)
        self.p_head = torch.nn.Linear(4, 1, bias=False)
        self.inputs = []

    def forward(self, input_ids, attention_mask, position_ids, **kwargs):
        self.inputs.append(
            {
                "input_ids": input_ids.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "position_ids": position_ids.detach().clone(),
                **kwargs,
            }
        )
        embedded = self.embedding(input_ids)
        weights = (~attention_mask[:, 0].isneginf()).to(embedded.dtype)
        hidden = embedded + weights @ embedded / weights.sum(-1, keepdim=True).clamp_min(1)
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=(hidden,))


class TinyQ(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 4)
        self.head = torch.nn.Linear(4, 1, bias=False)
        self.inputs = []

    def forward(self, input_ids, attention_mask, position_ids):
        self.inputs.append(
            {
                "input_ids": input_ids.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "position_ids": position_ids.detach().clone(),
            }
        )
        return self.head(self.embedding(input_ids)).squeeze(-1)


class RecordingContext:
    global_step = 1

    def __init__(self, main, q, accumulation_steps=1) -> None:
        self.main = main
        self.q = q
        self.accumulation_steps = accumulation_steps
        self.statistics = {}
        self.backward_calls = []
        self.reductions = []

    def forward_context(self):
        return nullcontext()

    def forward(self, inputs):
        return self.main(**dict(inputs), use_cache=False, return_dict=True)

    def forward_auxiliary(self, name, inputs):
        assert name == "q"
        return self.q(**dict(inputs))

    def module(self, path=""):
        return self.main.get_submodule(path)

    def auxiliary_model(self, name):
        assert name == "q"
        return self.q

    def backward(self, loss, *, weight=1.0):
        self.backward_calls.append((float(loss.detach()), weight))
        (loss * (weight / self.accumulation_steps)).backward()

    def reduce_statistics(self, values, *, reduction="sum", dtype="float32"):
        self.reductions.append((reduction, dtype, dict(values)))
        return dict(values)


def _trainer(divisor=4096.0):
    run = replace(RunConfig(), model=ModelConfig(attention="sdpa"), data=DataConfig(max_seq_len=6))
    config = SimpleNamespace(
        block_size=2,
        loss_backward_divisor=divisor,
        suffix_reconstruction_tau=8.0,
        lambda_align=1.0,
        lambda_recon=1.0,
    )
    trainer = SolDiffTrainer(run_config=run, recipe_config=config, data_factory=lambda **_: None)
    trainer.special_tokens = {"eos_token_id": 9, "pad_token_id": 0, "mask_token_id": 12}
    return trainer


def _batch():
    clean = torch.tensor([[1, 2, 9, 3, 4, 5]])
    return {"input_ids": clean, "labels": clean.clone(), "loss_mask": torch.ones_like(clean, dtype=torch.bool)}


def test_two_path_forward_updates_main_prior_and_independent_q() -> None:
    torch.manual_seed(81)
    main, q = TinyMain(), TinyQ()
    context = RecordingContext(main, q, accumulation_steps=2)
    trainer = _trainer()
    batch = _batch()
    torch.manual_seed(22)
    metrics = trainer.training_step(batch, context)
    assert len(main.inputs) == len(q.inputs) == 1
    assert q.inputs[0]["input_ids"].tolist() == batch["input_ids"].tolist()
    assert q.inputs[0]["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5]]
    combined = main.inputs[0]
    assert combined["input_ids"].shape == (2, 12)
    assert combined["input_ids"][:, 6:].tolist() == batch["input_ids"].repeat(2, 1).tolist()
    assert combined["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5] * 2] * 2
    assert combined["output_hidden_states"]
    assert not combined["output_router_logits"]
    assert not combined["use_cache"]
    assert "labels" not in combined
    assert context.backward_calls[0][1] == 1 / 4096
    assert metrics.values["loss"] == pytest.approx(context.backward_calls[0][0] / 4096)
    for module in (main.embedding, main.lm_head, main.p_head, q):
        grads = [parameter.grad for parameter in module.parameters()]
        assert all(value is not None and torch.isfinite(value).all() for value in grads)
        assert sum(float(value.abs().sum()) for value in grads) > 0
    # The statistic counts preserve both paths, independently of accumulation/divisor.
    assert context.statistics["soldiff"]["micro_batch_count"] == 1
    assert context.statistics["soldiff"]["group_count"] == 8
    reduced = trainer.finalize_step_metrics(metrics.values, context)
    assert reduced["loss"] == pytest.approx(metrics.values["loss"])
    assert reduced["q_model_norm"] > 0
    assert reduced["p_head_norm"] > 0


def test_fixed_divisor_is_separate_from_context_accumulation() -> None:
    torch.manual_seed(91)
    main, q = TinyMain(), TinyQ()
    full_context = RecordingContext(main, q, accumulation_steps=2)
    scaled_context = RecordingContext(deepcopy(main), deepcopy(q), accumulation_steps=2)
    torch.manual_seed(123)
    _trainer(1.0).training_step(_batch(), full_context)
    torch.manual_seed(123)
    _trainer(4096.0).training_step(_batch(), scaled_context)
    for full, scaled in ((full_context.main, scaled_context.main), (full_context.q, scaled_context.q)):
        for full_parameter, scaled_parameter in zip(full.parameters(), scaled.parameters(), strict=True):
            torch.testing.assert_close(full_parameter.grad / 4096, scaled_parameter.grad, rtol=2e-6, atol=1e-10)
    assert full_context.backward_calls[0][0] == scaled_context.backward_calls[0][0]


def test_native_qwen_main_and_q_run_the_complete_method(tmp_path, monkeypatch) -> None:
    from transformers import Qwen3Config

    import dfactory.models.qwen3.modeling as modeling
    from dfactory.models.qwen3 import Qwen3ForBDLM

    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    config = Qwen3Config(
        vocab_size=13,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        pad_token_id=None,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    q_config_path = tmp_path / "q_config.json"
    q_config_path.write_text(
        json.dumps(
            {
                "format": "mask_generator_q_linear.v1",
                "base_model_type": "qwen3",
                "encoder_num_layers": 1,
                "hidden_size": 16,
                "rms_eps": 1e-6,
                "q_score_bias": False,
                "q_head_type": "linear",
                "base_config": config.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    trainer = _trainer()
    trainer.run_config = replace(trainer.run_config, model=ModelConfig(attention="sdpa", init_device="cpu"))
    trainer.recipe_config.q_config_path = str(q_config_path)
    trainer.recipe_config.p_head_config_path = ""
    torch.manual_seed(421)
    main = trainer.prepare_main_model(Qwen3ForBDLM(config), None)
    auxiliary = trainer.build_auxiliary_models(main, None)
    trainer.after_parallelize(main, auxiliary)
    assert trainer._q_model is auxiliary["q"]
    context = RecordingContext(main, auxiliary["q"])
    torch.manual_seed(635)
    trainer.training_step(_batch(), context)
    for module in (main.model, main.lm_head, main.p_head, auxiliary["q"]):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(value is not None and torch.isfinite(value).all() for value in gradients)
        assert sum(float(value.abs().sum()) for value in gradients) > 0
    metrics = trainer.finalize_step_metrics({}, context)
    assert context.statistics["soldiff"]["group_count"] == 8
    assert metrics["loss"] > 0


def test_monitoring_uses_global_sums_counts_and_maxima_instead_of_mean_microbatch_ratios() -> None:
    trainer = _trainer()
    context = RecordingContext(TinyMain(), TinyQ())
    zero = torch.tensor(0.0)
    first = {
        "ce_sum": torch.tensor(4.0),
        "weighted_ce_sum": torch.tensor(2.0),
        "active_token_count": torch.tensor(1.0),
        "group_count": torch.tensor(2.0),
        "q_logits_sum": torch.tensor(2.0),
        "q_logits_sq_sum": torch.tensor(4.0),
        "q_logits_count": torch.tensor(1.0),
        "q_logits_abs_max": torch.tensor(2.0),
        "prior_logits_sum": torch.tensor(0.0),
        "prior_logits_sq_sum": torch.tensor(0.0),
        "prior_logits_count": torch.tensor(1.0),
        "prior_logits_abs_max": torch.tensor(0.0),
        "avg_rank_by_position_sum": torch.tensor([0.0, 2.0]),
        "avg_rank_by_position_count": torch.tensor([1.0, 1.0]),
        "inversion_norm_sum": torch.tensor(1.0),
    }
    second = {name: value.clone() for name, value in first.items()}
    second.update(
        {
            "ce_sum": torch.tensor(6.0),
            "active_token_count": torch.tensor(3.0),
            "q_logits_sum": torch.tensor(-2.0),
            "q_logits_abs_max": torch.tensor(1.0),
        }
    )
    trainer._accumulate_statistics(context, (torch.tensor(10.0), zero, zero, zero), first)
    trainer._accumulate_statistics(context, (torch.tensor(30.0), zero, zero, zero), second)
    metrics = trainer.finalize_step_metrics({}, context)
    assert metrics["ce_per_token"] == pytest.approx(10 / 4)  # Mean of 4/1 and 6/3 would incorrectly be 3.
    assert metrics["loss"] == 20 / 4096
    assert metrics["q_logits_std"] == 2  # Global moments, not a mean of local standard deviations.
    assert metrics["q_logits_abs_max"] == 2
    assert context.statistics["soldiff"]["group_count"] == 4
    assert metrics["order_stats/inversion_norm"] == 0.5
    assert metrics["order_stats/avg_rank_by_position/1"] == 2
    assert not {"kl_sum", "mean_abs_displacement_sum", "suffix_score_sum"} & context.statistics["soldiff"].keys()
    assert (
        not {
            "loss_unscaled",
            "kl_per_group",
            "active_token_count",
            "active_group_count",
            "phase_is_rloo",
            "loss_backward_divisor",
            "effective_loss_backward_divisor",
            "order_stats/mean_abs_displacement",
            "order_stats/suffix_score",
        }
        & metrics.keys()
    )
    assert [(reduction, dtype) for reduction, dtype, _ in context.reductions] == [
        ("sum", "float64"),
        ("sum", "float32"),
        ("max", "float32"),
    ]


@pytest.mark.parametrize(
    "bad_batch",
    [
        {"input_ids": torch.tensor([[1, 2]])},
        {"input_ids": torch.tensor([[1, 2, 9, 3, 4, 0]]), "loss_mask": torch.tensor([[1, 1, 1, 1, 1, 0]])},
    ],
)
def test_short_or_masked_training_blocks_are_rejected(bad_batch) -> None:
    with pytest.raises(ValueError, match="soldiff requires"):
        _trainer().training_step(bad_batch, RecordingContext(TinyMain(), TinyQ()))


class CPUGateway(VeOmniGateway):
    """Replace CUDA/sharding only; retain real optimizers, data, DCP and export."""

    def initialize(self, config, *, seed):
        torch.manual_seed(seed)
        return RuntimeContext("cpu", 0, 0, 1, 1, 0, 1, 1, 1)

    def parallelize(
        self, model, model_config, distributed_config, *, sharding_plan=None, weights_path="main", **kwargs
    ):
        from veomni.models.module_utils import load_model_weights

        if weights_path is None:
            model.to_empty(device="cpu")
            model.init_weights()
        else:
            load_model_weights(model, model_config.weights_path, "cpu")
        return model

    def activation_contexts(self, config):
        return nullcontext(), nullcontext()

    def synchronize(self):
        pass


def test_full_lifecycle_preserves_data_rng_main_p_q_and_resume_across_epochs(monkeypatch, tmp_path, capsys):
    import veomni.checkpoint.dcp_checkpointer as checkpoint_module

    import dfactory.models.qwen3.modeling as modeling
    import dfactory.runtime.veomni as runtime_module

    monkeypatch.setattr(runtime_module, "VeOmniGateway", CPUGateway)
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    monkeypatch.setattr(checkpoint_module, "empty_cache", lambda: None)
    monkeypatch.setattr(checkpoint_module, "synchronize", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    assets = tmp_path / "initial_model"
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        eos_token_id=14,
        pad_token_id=None,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(2026)
    Qwen3ForBDLM(config).save_pretrained(assets)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(16)}, unk_token="t0")),
        unk_token="t0",
        pad_token="t0",
        eos_token="t14",
        mask_token="t15",
    )
    tokenizer.save_pretrained(assets)
    q_config = tmp_path / "q.json"
    q_config.write_text(
        json.dumps({"base_config": config.to_dict(), "encoder_num_layers": 1, "rms_eps": 1e-6}), encoding="utf-8"
    )
    documents = tmp_path / "documents.jsonl"
    documents.write_text(
        "\n".join(
            json.dumps({"input_ids": [(index + offset) % 12 + 1 for offset in range(7)] + [14]}) for index in range(6)
        ),
        encoding="utf-8",
    )
    run = RunConfig(
        run=RunSettings(output_dir=str(tmp_path / "continuous"), seed=42, max_steps=5, num_train_epochs=2),
        recipe=RecipeConfig(
            name="soldiff", params={"block_size": 4, "q_config_path": str(q_config), "diagnostics_every_steps": 2}
        ),
        model=ModelConfig(
            provider="qwen3",
            config_path=str(assets),
            weights_path=str(assets),
            tokenizer_path=str(assets),
            attention="eager",
            mask_token_id=15,
            eos_token_id=14,
            pad_token_id=0,
        ),
        data=DataConfig(
            path=str(documents), format="token_records", text_key="input_ids", max_seq_len=8, num_workers=0
        ),
        optimization=OptimizationConfig(
            lr=0.001, min_lr=0.001, scheduler="constant", warmup_ratio=0, global_batch_size=2, micro_batch_size=1
        ),
        checkpoint=CheckpointConfig(save_every_steps=2, save_every_epochs=None, save_hf_weights=True),
        logging=LoggingConfig(report_to=["stdout"], log_every_steps=1),
    )
    spec = get_recipe("soldiff")
    seen = {}
    heads = {}

    def train(settings, name):
        trainer = spec.trainer_type(
            run_config=settings,
            recipe_config=spec.config_type.from_mapping(settings.recipe.params),
            data_factory=spec.data_factory,
        )
        original_step = trainer.training_step
        rows = seen[name] = []
        observed_models = []

        def observe(batch, context):
            if not rows:
                observed_models.append(context.module())
                if name == "resumed":
                    for key, value in context.module("p_head").state_dict().items():
                        torch.testing.assert_close(value, heads["interrupted"][key], rtol=0, atol=0)
            rows.append(batch["input_ids"].clone())
            return original_step(batch, context)

        trainer.training_step = observe
        trainer.train()
        heads[name] = {key: value.clone() for key, value in observed_models[0].p_head.state_dict().items()}

    train(run, "continuous")
    expected_next_rng = torch.rand(3)
    output = capsys.readouterr().out
    assert "[diagnostics step=2]" in output
    assert "[diagnostics step=4]" in output
    assert "[diagnostics step=1]" not in output
    records = [json.loads(line) for line in (tmp_path / "continuous/metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2, 3, 4, 5]
    assert all("q_model_norm" in record and "grad_norm" in record for record in records)
    assert all({"L_LO", "L_ord", "L_rec"} <= record.keys() for record in records)
    assert all(
        not {"local_rloo_loss", "prior_permutation_alignment_loss", "suffix_reconstruction_loss"} & record.keys()
        for record in records
    )
    assert all("loss_unscaled" not in record for record in records)
    interrupted = replace(
        run,
        run=replace(run.run, output_dir=str(tmp_path / "resumed"), max_steps=2),
        logging=LoggingConfig(report_to=[]),
    )
    train(interrupted, "interrupted")
    # Make the startup p sidecar deliberately conflict with the DCP tensors.
    # A full resume must overwrite this earlier load before its first forward.
    save_file(
        {key: torch.full_like(value, -10.0) for key, value in heads["interrupted"].items()},
        assets / "p_head.safetensors",
    )
    resumed = replace(
        interrupted,
        run=replace(interrupted.run, max_steps=5),
        checkpoint=replace(run.checkpoint, resume_from=str(tmp_path / "resumed/checkpoints/global_step_2")),
    )
    train(resumed, "resumed")
    actual_next_rng = torch.rand(3)
    assert not (tmp_path / "resumed/metrics.jsonl").exists()

    for actual, expected in zip(seen["interrupted"] + seen["resumed"], seen["continuous"], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for relative in ("model.safetensors", "q_model/model.safetensors"):
        expected = load_file(tmp_path / "continuous/checkpoints/global_step_5/hf_ckpt" / relative)
        actual = load_file(tmp_path / "resumed/checkpoints/global_step_5/hf_ckpt" / relative)
        assert expected.keys() == actual.keys()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)
    torch.testing.assert_close(actual_next_rng, expected_next_rng, rtol=0, atol=0)


def _logging_trainer(output_dir, *, block_size=32, diagnostics_every_steps=100):
    run = RunConfig()
    run = replace(run, run=replace(run.run, output_dir=str(output_dir)))
    return SolDiffTrainer(
        run_config=run,
        recipe_config=SolDiffConfig(
            q_config_path="q.json", block_size=block_size, diagnostics_every_steps=diagnostics_every_steps
        ),
        data_factory=lambda **_: None,
    )


def _metrics(block_size=32):
    values = {
        "loss": 21.125,
        "grad_norm": 4.5,
        "lr": 2e-5,
        "local_rloo_loss": 12.5,
        "prior_permutation_alignment_loss": 0.75,
        "suffix_reconstruction_loss": 7.875,
        "ce_per_token": 10.25,
        "q_weighted_ce_per_token": 10.5,
        "q_model_norm": 20.25,
        "p_head_norm": 4.125,
        "q_logits_std": 0.875,
        "q_logits_abs_max": 3.5,
        "p_logits_std": 0.125,
        "p_logits_abs_max": 0.75,
        "order_stats/inversion_norm": 0.5,
    }
    # Deliberately provide lexicographic input order (0, 1, 10, ..., 2, ...).
    for index in sorted(range(block_size), key=str):
        values[f"order_stats/avg_rank_by_position/{index}"] = index / 100
    return values


def test_normal_log_is_compact_but_jsonl_keeps_exact_selected_metrics(tmp_path, capsys):
    trainer = _logging_trainer(tmp_path)
    expected = _metrics()
    values = {
        **expected,
        "unused_metric": 123.0,
        "debug/count": 10.0,
    }
    original = deepcopy(values)
    trainer._stdout_metric_sink(1, values)
    assert capsys.readouterr().out.splitlines() == [
        "step=1 | loss=21.1250 | grad_norm=4.5000 | lr=2.00e-05",
        "  L_LO=12.5000 | L_ord=0.7500 | L_rec=7.8750 | ce/token=10.2500 | q_weighted_ce/token=10.5000",
    ]
    record = json.loads((tmp_path / "metrics.jsonl").read_text())
    expected_record = {"step": 1, **expected}
    expected_record["L_LO"] = expected_record.pop("local_rloo_loss")
    expected_record["L_ord"] = expected_record.pop("prior_permutation_alignment_loss")
    expected_record["L_rec"] = expected_record.pop("suffix_reconstruction_loss")
    assert record == expected_record
    assert values == original
    rank_indices = [int(name.rsplit("/", 1)[1]) for name in record if name.startswith("order_stats/avg_rank")]
    assert rank_indices == list(range(32))


def test_diagnostics_are_periodic_and_records_append_across_trainer_instances(tmp_path, capsys):
    values = _metrics()
    first = _logging_trainer(tmp_path)
    first._stdout_metric_sink(99, values)
    assert "diagnostics" not in capsys.readouterr().out
    first._stdout_metric_sink(100, values)
    diagnostics = capsys.readouterr().out
    assert "[diagnostics step=100]" in diagnostics
    assert "q_norm=20.2500 | p_norm=4.1250 | inversion=0.5000" in diagnostics
    assert (
        "q_logits_std=0.8750 | q_logits_abs_max=3.5000 | p_logits_std=0.1250 | p_logits_abs_max=0.7500" in diagnostics
    )
    second = _logging_trainer(tmp_path)
    second._stdout_metric_sink(101, values)
    assert "diagnostics" not in capsys.readouterr().out
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [99, 100, 101]
    assert all(record["q_model_norm"] == values["q_model_norm"] for record in records)
    for record in records:
        assert record["L_LO"] == values["local_rloo_loss"]
        assert record["L_ord"] == values["prior_permutation_alignment_loss"]
        assert record["L_rec"] == values["suffix_reconstruction_loss"]


@pytest.mark.parametrize("block_size", [2, 32, 36])
def test_position_diagnostics_use_numeric_order_and_groups_of_sixteen(block_size, tmp_path, capsys):
    trainer = _logging_trainer(tmp_path, block_size=block_size, diagnostics_every_steps=5)
    trainer._stdout_metric_sink(5, _metrics(block_size))
    lines = [line.strip() for line in capsys.readouterr().out.splitlines() if "order_avg_rank[" in line]
    assert len(lines) == (block_size + 15) // 16
    actual = []
    for start, line in zip(range(0, block_size, 16), lines, strict=True):
        end = min(start + 16, block_size)
        assert line.startswith(f"order_avg_rank[{start}:{end}] = ")
        actual.extend(json.loads(line.split(" = ", 1)[1]))
    assert actual == [index / 100 for index in range(block_size)]
