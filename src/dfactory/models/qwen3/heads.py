"""RMSNorm and gated MLP token scorer for SolDiff order learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers.activations import ACT2FN


def qwen_style_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)


class QwenStyleRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(hidden_size)))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return qwen_style_rms_norm(hidden_states, self.weight, self.variance_epsilon)


class TokenScoreMLP(nn.Module):
    """Token-level RMSNorm + gated MLP scorer with one scalar per token."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        rms_eps: float = 1e-6,
        activation: str = "silu",
        initializer_std: float = 0.02,
    ) -> None:
        super().__init__()
        if activation not in ACT2FN:
            raise ValueError(f"Unsupported activation {activation!r}.")
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.rms_eps = float(rms_eps)
        self.activation = str(activation)
        self.initializer_std = float(initializer_std)
        self.norm = QwenStyleRMSNorm(self.hidden_size, eps=self.rms_eps)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, 1, bias=False)
        self.act_fn = ACT2FN[self.activation]
        self.reset_parameters()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, QwenStyleRMSNorm):
            nn.init.ones_(module.weight)

    def reset_parameters(self) -> None:
        self.apply(self._init_weights)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        gated_hidden_states = self.act_fn(self.gate_proj(hidden_states))
        up_hidden_states = self.up_proj(hidden_states)
        return self.down_proj(gated_hidden_states * up_hidden_states).squeeze(-1)


@dataclass(frozen=True)
class PWarmupHeadConfig:
    hidden_size: int
    intermediate_size: int
    rms_eps: float
    activation: str
    initializer_std: float
    head_type: str = "rmsnorm_gated_mlp"

    def to_dict(self) -> dict[str, object]:
        return {
            "head_type": self.head_type,
            "hidden_size": int(self.hidden_size),
            "intermediate_size": int(self.intermediate_size),
            "rms_eps": float(self.rms_eps),
            "activation": self.activation,
            "initializer_std": float(self.initializer_std),
        }


def _first_parameter_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters(recurse=True):
        return parameter.device
    return torch.device("cpu")


def _get_decoder(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_decoder"):
        return model.get_decoder()
    if hasattr(model, "module"):
        return _get_decoder(model.module)
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "embed_tokens") and hasattr(model, "layers"):
        return model
    raise TypeError("Backbone model must expose get_decoder(), .module, or .model.")


def get_p_head(model: nn.Module) -> nn.Module:
    if hasattr(model, "p_head"):
        return model.p_head
    if hasattr(model, "module"):
        return get_p_head(model.module)
    raise AttributeError("Cannot find p_head on model.")


def build_p_head_config(
    model: nn.Module,
    *,
    intermediate_size: int | None = None,
    hidden_multiplier: float = 0.5,
    max_intermediate_size: int | None = 2048,
    rms_eps: float | None = None,
    activation: str | None = None,
) -> PWarmupHeadConfig:
    config = getattr(model, "config", None)
    if config is None:
        decoder = _get_decoder(model)
        config = getattr(decoder, "config", None)
    if config is None:
        raise ValueError("Cannot infer p_head config because model has no config.")

    hidden_size = int(getattr(config, "hidden_size"))
    if intermediate_size is None:
        intermediate_size = max(1, int(hidden_size * float(hidden_multiplier)))
        if max_intermediate_size is not None and int(max_intermediate_size) > 0:
            intermediate_size = min(int(max_intermediate_size), int(intermediate_size))
    if rms_eps is None:
        rms_eps = float(getattr(config, "rms_norm_eps", 1e-6))
    if activation is None:
        activation = str(getattr(config, "hidden_act", "silu"))
    initializer_std = float(getattr(config, "initializer_range", 0.02))
    return PWarmupHeadConfig(
        hidden_size=hidden_size,
        intermediate_size=int(intermediate_size),
        rms_eps=float(rms_eps),
        activation=str(activation),
        initializer_std=initializer_std,
    )


def attach_p_head(
    model: nn.Module,
    *,
    intermediate_size: int | None = None,
    hidden_multiplier: float = 0.5,
    max_intermediate_size: int | None = 2048,
    rms_eps: float | None = None,
    activation: str | None = None,
    replace_existing: bool = False,
    p_head_config_path: str | Path | None = None,
    config_values: dict[str, Any] | None = None,
) -> PWarmupHeadConfig:
    values = dict(config_values or {})
    if p_head_config_path:
        with Path(p_head_config_path).open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("p head config must be a JSON object")
        values.update(loaded)
    if values.get("head_type", "rmsnorm_gated_mlp") != "rmsnorm_gated_mlp":
        raise ValueError("p head config head_type must be rmsnorm_gated_mlp")
    hidden_size = int(getattr(model.config, "hidden_size"))
    if "hidden_size" in values and int(values["hidden_size"]) != hidden_size:
        raise ValueError("p head hidden_size must match the main model")
    intermediate_size = (
        int(values.get("intermediate_size", intermediate_size))
        if (values.get("intermediate_size", intermediate_size) is not None)
        else None
    )
    rms_eps = float(values.get("rms_eps", rms_eps)) if values.get("rms_eps", rms_eps) is not None else None
    activation = str(values.get("activation", activation)) if values.get("activation", activation) is not None else None
    if hasattr(model, "p_head") and not replace_existing:
        # A restored model already reconstructed its head from the main config.
        existing = dict(getattr(model.config, "p_head_config", {}) or {})
        if values and any(existing.get(key) != value for key, value in values.items() if key in existing):
            raise ValueError("p head configuration conflicts with the restored main model")
        if existing:
            return PWarmupHeadConfig(**existing)
        raise ValueError("Model already has p_head; pass replace_existing=True to replace it.")
    head_config = build_p_head_config(
        model,
        intermediate_size=intermediate_size,
        hidden_multiplier=hidden_multiplier,
        max_intermediate_size=max_intermediate_size,
        rms_eps=rms_eps,
        activation=activation,
    )
    if head_config.intermediate_size <= 0 or head_config.rms_eps <= 0:
        raise ValueError("p head intermediate_size and rms_eps must be positive")
    if "initializer_std" in values:
        from dataclasses import replace

        head_config = replace(head_config, initializer_std=float(values["initializer_std"]))
    device = _first_parameter_device(model)
    with torch.device(device):
        p_head = TokenScoreMLP(
            head_config.hidden_size,
            head_config.intermediate_size,
            rms_eps=head_config.rms_eps,
            activation=head_config.activation,
            initializer_std=head_config.initializer_std,
        )
    setattr(model, "p_head", p_head)
    model.config.p_head_config = head_config.to_dict()
    no_split = list(getattr(model, "_no_split_modules", []) or [])
    if TokenScoreMLP.__name__ not in no_split:
        no_split.append(TokenScoreMLP.__name__)
    model._no_split_modules = no_split
    return head_config


__all__ = ["PWarmupHeadConfig", "QwenStyleRMSNorm", "TokenScoreMLP", "attach_p_head", "get_p_head"]
