"""Self-contained trainable q prefix encoder and token scorer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import Qwen3Config

from dfactory.models.qwen3 import Qwen3Model
from dfactory.models.qwen3.heads import QwenStyleRMSNorm
from dfactory.recipes.soldiff.q_config import load_q_config


class TeacherPrefixEncoder(nn.Module):
    """Embeddings and the first L transformer decoder blocks."""

    def __init__(self, decoder: nn.Module, num_layers: int) -> None:
        super().__init__()
        if not hasattr(decoder, "embed_tokens") or not hasattr(decoder, "layers"):
            raise TypeError("Teacher decoder must expose embed_tokens and layers.")
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive.")
        if int(num_layers) > len(decoder.layers):
            raise ValueError(f"Requested {num_layers} layers but decoder only has {len(decoder.layers)}.")

        self.embed_tokens = decoder.embed_tokens
        self.layers = nn.ModuleList(list(decoder.layers[: int(num_layers)]))
        self.rotary_emb = getattr(decoder, "rotary_emb", None)
        self.num_layers = int(num_layers)
        self.config = getattr(decoder, "config", None)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)
        cache_position = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)

        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
        return hidden_states


class TokenScoreLinearHead(nn.Module):
    """Qwen-style RMSNorm + linear token scorer with one scalar per token."""

    def __init__(self, hidden_size: int, *, rms_eps: float = 1e-6, bias: bool = False) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rms_eps = float(rms_eps)
        self.bias = bool(bias)
        self.norm = QwenStyleRMSNorm(hidden_size=int(hidden_size), eps=float(rms_eps))
        self.score = nn.Linear(int(hidden_size), 1, bias=bool(bias))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.score(self.norm(hidden_states)).squeeze(-1)


class CheckpointableTeacherPrefixEncoder(TeacherPrefixEncoder):
    def __init__(self, decoder: nn.Module, num_layers: int) -> None:
        super().__init__(decoder, num_layers)
        self.gradient_checkpointing = False
        self.gradient_checkpointing_kwargs: dict[str, Any] = {"use_reentrant": False}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        self.gradient_checkpointing = True
        self.gradient_checkpointing_kwargs = dict(gradient_checkpointing_kwargs or {"use_reentrant": False})
        if (
            self.gradient_checkpointing_kwargs.get("use_reentrant")
            and "context_fn" in self.gradient_checkpointing_kwargs
        ):
            self.gradient_checkpointing_kwargs.pop("context_fn")

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)
        cache_position = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)

        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:

                def layer_forward(states, layer=layer):
                    layer_outputs = layer(
                        states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                    return layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer_forward,
                    hidden_states,
                    **self.gradient_checkpointing_kwargs,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        return hidden_states


class MaskGeneratorQLinear(nn.Module):
    _no_split_modules = ["Qwen3DecoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, prefix_encoder: TeacherPrefixEncoder, q_head: TokenScoreLinearHead) -> None:
        super().__init__()
        self.prefix_encoder = prefix_encoder
        self.q_head = q_head
        self.gradient_checkpointing = False
        self.config = SimpleNamespace(
            architectures=["MaskGeneratorQLinear"],
            model_type="mask_generator_q_linear",
            tie_word_embeddings=False,
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, QwenStyleRMSNorm):
            nn.init.ones_(module.weight)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        if not hasattr(self.prefix_encoder, "gradient_checkpointing_enable"):
            raise RuntimeError("prefix_encoder does not support gradient checkpointing.")
        self.gradient_checkpointing = True
        self.prefix_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False
        if hasattr(self.prefix_encoder, "gradient_checkpointing_disable"):
            self.prefix_encoder.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.prefix_encoder(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        return self.q_head(hidden_states)


def attach_random_q_init_weights(q_model):
    def _initializer_range() -> float:
        configs = (
            getattr(q_model, "config", None),
            getattr(getattr(q_model, "prefix_encoder", None), "config", None),
        )
        for config in configs:
            value = getattr(config, "initializer_range", None)
            if value is not None:
                return float(value)
        return 0.02

    def init_weights() -> None:
        initializer_range = _initializer_range()

        @torch.no_grad()
        def init_module(module) -> None:
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, torch.nn.LayerNorm) or module.__class__.__name__.endswith("RMSNorm"):
                weight = getattr(module, "weight", None)
                if weight is not None:
                    weight.data.fill_(1.0)
                bias = getattr(module, "bias", None)
                if bias is not None:
                    bias.data.zero_()
            rope_init_fn = getattr(module, "rope_init_fn", None)
            if rope_init_fn is not None and hasattr(module, "config") and hasattr(module, "inv_freq"):
                inv_freq, attention_scaling = rope_init_fn(module.config, module.inv_freq.device)
                module.attention_scaling = attention_scaling
                module.register_buffer("inv_freq", inv_freq, persistent=False)
                module.original_inv_freq = module.inv_freq

        q_model.apply(init_module)

    q_model.init_weights = init_weights
    return q_model


def build_q_model(
    q_model_path: str | Path,
    *,
    init_device: str = "meta",
    torch_dtype: torch.dtype | str = torch.float32,
    attn_implementation: str = "flex_attention",
) -> MaskGeneratorQLinear:
    """Build a fresh q from its architecture; package weights are never read here."""
    package_config = load_q_config(q_model_path)
    base_config = Qwen3Config.from_dict(package_config["base_config"])
    base_config._attn_implementation = attn_implementation
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype.removeprefix("torch."))
    with torch.device(init_device):
        decoder = Qwen3Model(base_config)
        prefix_encoder = CheckpointableTeacherPrefixEncoder(
            decoder,
            num_layers=package_config["encoder_num_layers"],
        )
        q_head = TokenScoreLinearHead(
            base_config.hidden_size,
            rms_eps=package_config["rms_eps"],
            bias=package_config["q_score_bias"],
        )
        model = MaskGeneratorQLinear(prefix_encoder, q_head)
    model = model.to(dtype=torch_dtype)
    model._q_package_config = package_config
    attach_random_q_init_weights(model)
    if init_device != "meta":
        model.init_weights()
    model.requires_grad_(True)
    model.train()
    return model


__all__ = [
    "MaskGeneratorQLinear",
    "CheckpointableTeacherPrefixEncoder",
    "TeacherPrefixEncoder",
    "TokenScoreLinearHead",
    "attach_random_q_init_weights",
    "build_q_model",
    "load_q_config",
]
