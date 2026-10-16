"""dFactory-owned loading and asset semantics for LLaDA2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from dfactory.config import ModelConfig
from dfactory.models.llada2.configuration import LLaDA2MoeConfig
from dfactory.models.llada2.modeling import LLaDA2MoeModelLM
from dfactory.models.llada2.parallel import get_parallel_plan


class LLaDA2Provider:
    def load_tokenizer(self, config: ModelConfig) -> Any:
        return AutoTokenizer.from_pretrained(config.tokenizer_path, padding_side="right", trust_remote_code=True)

    def build_model(self, config: ModelConfig) -> LLaDA2MoeModelLM:
        model_config = LLaDA2MoeConfig.from_pretrained(config.config_path)
        model_config._attn_implementation = config.attention
        model_config._moe_implementation = config.moe
        for name in ("mask", "eos", "pad"):
            value = getattr(config, f"{name}_token_id")
            if value is not None:
                setattr(model_config, f"{name}_token_id", value)
        with torch.device(config.init_device):
            return LLaDA2MoeModelLM(model_config)

    def parallel_policy(self, model: Any) -> Any:
        del model
        return get_parallel_plan()

    def special_tokens(self, tokenizer: Any, model_config: Any, config: ModelConfig) -> dict[str, int]:
        resolved: dict[str, int] = {}
        for name in ("mask", "eos", "pad"):
            explicit = getattr(config, f"{name}_token_id")
            tokenizer_value = getattr(tokenizer, f"{name}_token_id", None)
            model_value = getattr(model_config, f"{name}_token_id", None)
            value = explicit if explicit is not None else tokenizer_value
            value = model_value if value is None else value
            if value is None:
                raise ValueError(f"LLaDA2 requires a {name}_token_id from model config, tokenizer, or run config")
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
        config_path = target / "config.json"
        exported_config = json.loads(config_path.read_text(encoding="utf-8"))
        exported_config["auto_map"] = {
            "AutoConfig": "configuration_dfactory_llada2.LLaDA2MoeConfig",
            "AutoModel": "modeling_dfactory_llada2.LLaDA2MoeModel",
            "AutoModelForCausalLM": "modeling_dfactory_llada2.LLaDA2MoeModelLM",
        }
        config_path.write_text(json.dumps(exported_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (target / "configuration_dfactory_llada2.py").write_text(
            'from dfactory.models.llada2 import LLaDA2MoeConfig\n\n__all__ = ["LLaDA2MoeConfig"]\n',
            encoding="utf-8",
        )
        (target / "modeling_dfactory_llada2.py").write_text(
            "from dfactory.models.llada2 import LLaDA2MoeModel, LLaDA2MoeModelLM\n\n"
            '__all__ = ["LLaDA2MoeModel", "LLaDA2MoeModelLM"]\n',
            encoding="utf-8",
        )


__all__ = ["LLaDA2Provider"]
