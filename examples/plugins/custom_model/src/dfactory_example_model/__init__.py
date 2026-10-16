"""Installable Hugging Face causal language-model provider example."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dfactory import ModelSpec


class HuggingFaceCausalLMProvider:
    def load_tokenizer(self, config: Any) -> Any:
        return AutoTokenizer.from_pretrained(config.tokenizer_path, padding_side="right")

    def build_model(self, config: Any) -> Any:
        model_config = AutoConfig.from_pretrained(config.config_path)
        model_config._attn_implementation = config.attention
        with torch.device(config.init_device):
            return AutoModelForCausalLM.from_config(model_config)

    def parallel_policy(self, model: Any) -> None:
        del model
        return None

    def special_tokens(self, tokenizer: Any, model_config: Any, config: Any) -> dict[str, int]:
        resolved: dict[str, int] = {}
        for name in ("eos", "pad"):
            value = getattr(config, f"{name}_token_id", None)
            if value is None:
                value = getattr(tokenizer, f"{name}_token_id", None)
            if value is None:
                value = getattr(model_config, f"{name}_token_id", None)
            if value is None:
                raise ValueError(f"hf_causal_lm requires {name}_token_id")
            resolved[f"{name}_token_id"] = int(value)
        return resolved

    def save_assets(self, model: Any, tokenizer: Any, output_dir: str | Path) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        model.config.save_pretrained(target)
        tokenizer.save_pretrained(target)

    def save_weights(
        self,
        model: Any,
        tokenizer: Any,
        state_dict: dict[str, Any],
        output_dir: str | Path,
    ) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(target, state_dict=state_dict, safe_serialization=True)
        tokenizer.save_pretrained(target)


model_spec = ModelSpec(
    name="hf_causal_lm",
    provider_type=HuggingFaceCausalLMProvider,
    description="Generic AutoModelForCausalLM provider implemented as an external plugin.",
)

__all__ = ["HuggingFaceCausalLMProvider", "model_spec"]
