"""SolDiff token rows, configuration validation and model assets."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DistributedSampler

from dfactory.config import DataConfig, DistributedConfig, ModelConfig, RunConfig, load_config
from dfactory.data.collators import collate_train_batch
from dfactory.data.sources.jsonl import JsonlRecordSource
from dfactory.recipes.soldiff.config import SolDiffConfig
from dfactory.recipes.soldiff.data import FixedTokenRows, build_data, validate_run


def test_documents_fill_rows_without_padding_or_duplicate_eos() -> None:
    dataset = FixedTokenRows(
        [{"input_ids": [1, 2]}, {"input_ids": [3, 9]}, {"input_ids": [4, 5, 6]}],
        max_seq_len=4,
        eos_token_id=9,
        token_key="input_ids",
    )
    assert dataset.total_tokens == 9
    assert dataset.inserted_eos_tokens == 2
    assert dataset.discarded_tail_tokens == 1
    assert len(dataset) == 2
    assert dataset[0]["input_ids"].tolist() == [1, 2, 9, 3]
    assert dataset[1]["input_ids"].tolist() == [9, 4, 5, 6]
    batch = collate_train_batch([dataset[0], dataset[1]])
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert batch["loss_mask"].all()
    assert torch.equal(batch["labels"], batch["input_ids"])
    assert batch["attention_spec"]["sequence_lengths"].tolist() == [[4], [4]]
    assert batch["attention_spec"]["valid_lengths"].tolist() == [[4], [4]]
    assert "noisy_input_ids" not in batch


def test_long_document_continues_across_rows_without_per_document_truncation() -> None:
    dataset = FixedTokenRows(
        [{"ids": [1, 2, 3, 4, 5, 6, 7, 8, 9]}, {"ids": [10, 99]}],
        max_seq_len=4,
        eos_token_id=99,
        token_key="ids",
    )
    assert len(dataset) == 3
    assert dataset.discarded_tail_tokens == 0
    assert [dataset[index]["input_ids"].tolist() for index in range(3)] == [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 99, 10, 99],
    ]


def test_internal_eos_and_source_tokens_are_preserved() -> None:
    dataset = FixedTokenRows([{"input_ids": [1, 9, 2, 9]}], max_seq_len=4, eos_token_id=9, token_key="input_ids")
    assert dataset.inserted_eos_tokens == 0
    assert dataset[0]["input_ids"].tolist() == [1, 9, 2, 9]


def test_plan_is_global_and_distributed_sampler_partitions_complete_rows() -> None:
    source = [{"input_ids": [value, 99]} for value in range(8)]
    first = FixedTokenRows(source, max_seq_len=4, eos_token_id=99, token_key="input_ids")
    second = FixedTokenRows(source, max_seq_len=4, eos_token_id=99, token_key="input_ids")
    assert len(first) == len(second) == 4
    samplers = [
        DistributedSampler(dataset, num_replicas=2, rank=rank, seed=42) for rank, dataset in enumerate([first, second])
    ]
    indices = [list(sampler) for sampler in samplers]
    assert set(indices[0]).isdisjoint(indices[1])
    assert set(indices[0] + indices[1]) == set(range(4))
    assert [first[index]["input_ids"].tolist() for index in range(4)] == [
        second[index]["input_ids"].tolist() for index in range(4)
    ]
    samplers[0].set_epoch(3)
    resumed_sampler = DistributedSampler(first, num_replicas=2, rank=0, seed=42)
    resumed_sampler.set_epoch(3)
    assert list(samplers[0]) == list(resumed_sampler)


def test_jsonl_source_builds_clean_rows_and_rejects_second_packing(tmp_path) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text('{"input_ids":[1,2,3]}\n{"input_ids":[4,5,6]}\n', encoding="utf-8")
    data_config = DataConfig(format="token_records", text_key="input_ids", max_seq_len=4, packing=False)
    config = SolDiffConfig(block_size=2, q_config_path="unused-for-data.json")
    data = build_data(
        source=JsonlRecordSource(path),
        tokenizer=None,
        data_config=data_config,
        recipe_config=config,
        special_tokens={"eos_token_id": 9},
    )
    assert data.dataset[0]["input_ids"].tolist() == [1, 2, 3, 9]
    assert data.dataset[1]["input_ids"].tolist() == [4, 5, 6, 9]
    with pytest.raises(ValueError, match="packing=false"):
        build_data(
            source=[],
            tokenizer=None,
            data_config=replace(data_config, packing=True),
            recipe_config=config,
            special_tokens={"eos_token_id": 9},
        )


@pytest.mark.parametrize("tokens", [[], [True], [-1], [1.5], "text", [[1]], torch.tensor([1.0])])
def test_invalid_document_tokens_fail_before_row_sampling(tokens: object) -> None:
    with pytest.raises(ValueError, match="token record"):
        FixedTokenRows([{"input_ids": tokens}], max_seq_len=4, eos_token_id=9, token_key="input_ids")


def test_insufficient_source_fails_instead_of_padding() -> None:
    with pytest.raises(ValueError, match="at least one complete row"):
        FixedTokenRows([{"input_ids": [1]}], max_seq_len=4, eos_token_id=9, token_key="input_ids")


def test_source_mutation_invalidates_the_row_plan() -> None:
    source = [{"input_ids": [1, 2, 3]}]
    dataset = FixedTokenRows(source, max_seq_len=4, eos_token_id=9, token_key="input_ids")
    source[0]["input_ids"].append(4)
    with pytest.raises(ValueError, match="changed after row planning"):
        dataset[0]


def test_map_indices_do_not_wrap_or_return_incomplete_rows() -> None:
    dataset = FixedTokenRows([{"input_ids": [1, 2, 3]}], max_seq_len=4, eos_token_id=9, token_key="input_ids")
    for index in (-1, 1, True):
        with pytest.raises(IndexError, match="out of range"):
            dataset[index]


def _q_config(tmp_path: Path) -> str:
    path = tmp_path / "q_config.json"
    path.write_text(
        json.dumps(
            {
                "encoder_num_layers": 1,
                "rms_eps": 1e-6,
                "base_config": {
                    "model_type": "qwen3",
                    "vocab_size": 32,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "max_position_embeddings": 64,
                },
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _run() -> RunConfig:
    return RunConfig(
        model=ModelConfig(provider="qwen3"),
        data=DataConfig(format="token_records", text_key="input_ids", max_seq_len=64, packing=False),
        distributed=DistributedConfig(ep=1),
    )


def test_config_requires_explicit_q_architecture_and_rejects_unsupported_modes() -> None:
    with pytest.raises(ValueError, match="q_config_path"):
        SolDiffConfig.from_mapping({})
    for name in ("doc_level", "block_diffusion_mode", "same_token_labels", "enable_block_internal_rloo"):
        with pytest.raises(ValueError, match="unknown soldiff parameter"):
            SolDiffConfig.from_mapping({"q_config_path": "q.json", name: False})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("block_size", True),
        ("block_size", 0),
        ("block_size", 2.5),
        ("loss_backward_divisor", 0),
        ("loss_backward_divisor", float("nan")),
        ("suffix_reconstruction_tau", -1),
        ("suffix_reconstruction_tau", float("inf")),
        ("lambda_align", False),
        ("lambda_recon", float("nan")),
        ("p_head_config_path", 3),
        ("diagnostics_every_steps", 0),
        ("diagnostics_every_steps", True),
        ("diagnostics_every_steps", 1.5),
    ],
)
def test_config_rejects_invalid_algorithm_values(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        SolDiffConfig.from_mapping({"q_config_path": "q.json", name: value})


def test_loss_weights_accept_finite_signed_values() -> None:
    config = SolDiffConfig.from_mapping({"q_config_path": "q.json", "lambda_align": -1.0, "lambda_recon": 0.0})
    assert config.lambda_align == -1.0
    assert config.lambda_recon == 0.0


def test_validate_run_checks_real_q_config_and_method_boundaries(tmp_path: Path) -> None:
    config = SolDiffConfig(q_config_path=_q_config(tmp_path))
    run = _run()
    validate_run(recipe_config=config, run_config=run)
    cases = [
        (replace(run, model=replace(run.model, provider="llada2")), "model.provider=qwen3"),
        (replace(run, data=replace(run.data, format="conversation")), "token_records"),
        (replace(run, data=replace(run.data, packing=True)), "packing=false"),
        (replace(run, data=replace(run.data, max_seq_len=63)), "divisible"),
        (replace(run, distributed=replace(run.distributed, ep=2)), "distributed.ep=1"),
    ]
    for invalid_run, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_run(recipe_config=config, run_config=invalid_run)


def test_validate_run_rejects_missing_head_config(tmp_path: Path) -> None:
    config = SolDiffConfig(q_config_path=_q_config(tmp_path), p_head_config_path=str(tmp_path / "missing-head.json"))
    with pytest.raises(ValueError, match="p_head_config_path"):
        validate_run(recipe_config=config, run_config=_run())


@pytest.mark.parametrize("profile", ["1.7b", "8b"])
def test_bundled_recipe_configuration_validates(profile, monkeypatch) -> None:
    root = Path(__file__).parents[3]
    monkeypatch.chdir(root)
    path = root / "configs" / "recipes" / "soldiff" / f"qwen3_{profile}.yaml"
    run = load_config(path)
    validate_run(recipe_config=SolDiffConfig.from_mapping(run.recipe.params), run_config=run)


@pytest.mark.parametrize(("profile", "hidden_size", "layers"), [("1.7b", 2048, 28), ("8b", 4096, 36)])
def test_bundled_qwen_assets_have_expected_model_and_tokenizer_settings(profile, hidden_size, layers) -> None:
    from transformers import AutoTokenizer

    root = Path(__file__).parents[3]
    tokenizer = AutoTokenizer.from_pretrained(root / "configs" / "tokenizer" / "qwen3_tokenizer", local_files_only=True)
    assert tokenizer.eos_token_id == 151645
    assert tokenizer.pad_token_id == 151643
    assert tokenizer.convert_tokens_to_ids("<MASK>") == 151669
    model_config = json.loads((root / "configs" / "model_configs" / f"qwen3_{profile}" / "config.json").read_text())
    assert model_config["hidden_size"] == hidden_size
    assert model_config["num_hidden_layers"] == layers
    assert model_config["vocab_size"] > tokenizer.convert_tokens_to_ids("<MASK>")
