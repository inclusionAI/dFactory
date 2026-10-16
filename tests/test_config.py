from pathlib import Path

import pytest

import dfactory.recipes
from dfactory.config import ConfigError, load_config

VALID_CONFIG = """
run:
  output_dir: outputs/test
  max_steps: 2
recipe:
  name: sft
  params:
    block_size: 4
model:
  provider: llada2
  config_path: configs/model_configs/llada2_mini
  weights_path: /models/llada2
  tokenizer_path: /models/llada2
  attention: sdpa
  moe: fused
  init_device: meta
data:
  source: jsonl
  path: data/train.jsonl
  max_seq_len: 16
distributed:
  mode: fsdp2
optimization:
  optimizer: adamw
  scheduler: cosine
  global_batch_size: 2
  micro_batch_size: 1
checkpoint: {}
logging: {}
"""

BUNDLED_CONFIGS = sorted((Path(__file__).parents[1] / "configs" / "recipes").glob("*/*.yaml"))


def _write_config(tmp_path: Path, text: str = VALID_CONFIG) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("config_path", BUNDLED_CONFIGS, ids=lambda path: path.parent.name)
def test_bundled_recipe_config_is_valid(config_path: Path) -> None:
    config = load_config(config_path)

    assert config.distributed.dp >= 2
    assert config.optimization.global_batch_size % (config.optimization.micro_batch_size * config.distributed.dp) == 0


def test_loads_strict_config_and_applies_overrides(tmp_path: Path) -> None:
    config = load_config(
        _write_config(tmp_path),
        overrides=["run.max_steps=4", "recipe.params.noise_range=[0.2, 0.6]"],
    )
    assert config.run.max_steps == 4
    assert config.recipe.params["noise_range"] == [0.2, 0.6]


def test_rejects_unknown_section(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown config section"):
        load_config(_write_config(tmp_path, VALID_CONFIG + "internal_cluster: {}\n"))


def test_rejects_unknown_field(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown field.*model"):
        load_config(
            _write_config(tmp_path, VALID_CONFIG.replace("  provider: llada2", "  provider: llada2\n  typo: true"))
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("model.attention=flash_attention_2", "model.attention must be one of"),
        ("model.moe=unknown", "model.moe must be one of"),
        ("model.init_device=cpu", "model.init_device must be meta for FSDP2"),
        ("optimization.optimizer=sgd", "optimization.optimizer must be one of"),
        ("optimization.scheduler=polynomial", "optimization.scheduler must be one of"),
        ("logging.report_to=[wandb]", "logging.report_to may contain only: stdout"),
    ],
)
def test_rejects_invalid_public_enum_values(tmp_path: Path, override: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(_write_config(tmp_path), overrides=[override])


def test_rejects_unknown_recipe_parameter(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown SFT parameter"):
        load_config(
            _write_config(tmp_path, VALID_CONFIG.replace("    block_size: 4", "    block_size: 4\n    typo: true"))
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("  provider: llada2", "  provider: missing-model", "provider 'missing-model' was not found"),
        ("  source: jsonl", "  source: missing-source", "provider 'missing-source' was not found"),
    ],
)
def test_rejects_unavailable_selected_provider(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(_write_config(tmp_path, VALID_CONFIG.replace(old, new)))


def test_rejects_sequence_length_not_aligned_to_recipe_block(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be divisible"):
        load_config(_write_config(tmp_path, VALID_CONFIG.replace("  max_seq_len: 16", "  max_seq_len: 15")))


def test_recipe_without_block_size_owns_its_validation(monkeypatch, tmp_path: Path) -> None:
    class ConfigWithoutBlockSize:
        @classmethod
        def from_mapping(cls, values):
            assert values == {}
            return cls()

    recipe_spec = type(
        "RecipeSpec",
        (),
        {
            "config_type": ConfigWithoutBlockSize,
            "validate_run": staticmethod(lambda **kwargs: None),
        },
    )()
    monkeypatch.setattr(dfactory.recipes, "get_recipe", lambda name: recipe_spec)
    config_text = VALID_CONFIG.replace("name: sft", "name: external").replace(
        "  params:\n    block_size: 4\n", "  params: {}\n"
    )

    config = load_config(_write_config(tmp_path, config_text))

    assert config.recipe.name == "external"


@pytest.mark.parametrize(
    ("distributed", "message"),
    [
        ("  mode: ddp\n", "distributed.mode must be fsdp2"),
        ("  mode: fsdp2\n  tp: 2\n", "unknown field.*distributed"),
        ("  mode: fsdp2\n  sp: 2\n", "unknown field.*distributed"),
    ],
)
def test_rejects_unimplemented_parallel_modes(tmp_path: Path, distributed: str, message: str) -> None:
    config_text = VALID_CONFIG.replace("  mode: fsdp2\n", distributed)
    with pytest.raises(ConfigError, match=message):
        load_config(_write_config(tmp_path, config_text))


def test_rejects_removed_fsdp_cpu_offload_setting(tmp_path: Path) -> None:
    config_text = VALID_CONFIG.replace("  mode: fsdp2\n", "  mode: fsdp2\n  fsdp_offload: true\n")

    with pytest.raises(ConfigError, match="unknown field.*distributed"):
        load_config(_write_config(tmp_path, config_text))


def test_rejects_single_gpu_training_configuration(tmp_path: Path) -> None:
    config_text = VALID_CONFIG.replace("  mode: fsdp2\n", "  mode: fsdp2\n  dp: 1\n")

    with pytest.raises(ConfigError, match="distributed.dp must be at least 2"):
        load_config(_write_config(tmp_path, config_text))


def test_rejects_eager_moe_with_expert_parallelism(tmp_path: Path) -> None:
    config_text = VALID_CONFIG.replace("  moe: fused\n", "  moe: eager\n").replace(
        "  mode: fsdp2\n", "  mode: fsdp2\n  dp: 2\n  ep: 2\n"
    )

    with pytest.raises(ConfigError, match="model.moe=eager does not support distributed.ep > 1"):
        load_config(_write_config(tmp_path, config_text))


def test_rejects_global_batch_that_is_not_divisible_by_data_parallel_batch(tmp_path: Path) -> None:
    config_text = VALID_CONFIG.replace("  global_batch_size: 2\n", "  global_batch_size: 1\n")

    with pytest.raises(ConfigError, match=r"global_batch_size must be divisible by micro_batch_size \* distributed.dp"):
        load_config(_write_config(tmp_path, config_text))
