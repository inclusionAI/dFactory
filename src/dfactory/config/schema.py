"""Typed configuration owned by dFactory, independent of VeOmni arguments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class RunSettings:
    output_dir: str = "outputs"
    seed: int = 42
    max_steps: int = 1_000
    num_train_epochs: int = 1


@dataclass(frozen=True)
class RecipeConfig:
    name: str = "sft"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "llada2"
    config_path: str = ""
    weights_path: str = ""
    tokenizer_path: str = ""
    attention: Literal["eager", "sdpa", "flex_attention"] = "sdpa"
    moe: Literal["fused", "eager"] = "fused"
    init_device: Literal["meta"] = "meta"
    mask_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None


@dataclass(frozen=True)
class DataConfig:
    source: str = "jsonl"
    path: str = ""
    format: Literal["conversation", "token_records"] = "conversation"
    text_key: str = "messages"
    max_seq_len: int = 2_048
    packing: bool = False
    num_workers: int = 0


@dataclass(frozen=True)
class DistributedConfig:
    mode: Literal["fsdp2"] = "fsdp2"
    dp: int = 2
    dp_replicate: int = 1
    ep: int = 1
    activation_offload: bool = False
    gradient_checkpointing: bool = True


@dataclass(frozen=True)
class OptimizationConfig:
    optimizer: Literal["adamw", "anyprecision_adamw"] = "adamw"
    lr: float = 1.0e-5
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.1
    scheduler: Literal["constant", "linear", "cosine"] = "cosine"
    warmup_ratio: float = 0.03
    decay_ratio: float = 1.0
    min_lr: float = 1.0e-7
    global_batch_size: int = 8
    micro_batch_size: int = 1
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class CheckpointConfig:
    resume_from: str | None = None
    save_every_steps: int | None = None
    save_every_epochs: int | None = 1
    save_hf_weights: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    report_to: list[Literal["stdout"]] = field(default_factory=lambda: ["stdout"])
    log_every_steps: int = 10


@dataclass(frozen=True)
class RunConfig:
    run: RunSettings = field(default_factory=RunSettings)
    recipe: RecipeConfig = field(default_factory=RecipeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str, overrides: tuple[str, ...] | list[str] = ()) -> RunConfig:
        from dfactory.config.loader import load_config

        return load_config(path, overrides=overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)
