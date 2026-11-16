"""Native Qwen3 configuration and Block Diffusion checkpoint assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoTokenizer, Qwen3Config

from dfactory.config import ModelConfig

if TYPE_CHECKING:
    from dfactory.models.qwen3.modeling import Qwen3ForBDLM


class Qwen3Provider:
    def load_tokenizer(self, config: ModelConfig) -> Any:
        return AutoTokenizer.from_pretrained(config.tokenizer_path, padding_side="right", trust_remote_code=True)

    def build_model(self, config: ModelConfig) -> Qwen3ForBDLM:
        from transformers.modeling_utils import no_init_weights

        from dfactory.models.qwen3.modeling import Qwen3ForBDLM
        from dfactory.runtime.veomni.ops import init_empty_weights

        model_config = Qwen3Config.from_pretrained(config.config_path)
        model_config._attn_implementation = config.attention
        # Run EOS/PAD hints describe data boundaries and fill tokens. They must not
        # change native embedding.padding_idx or the backbone's native config.
        if config.mask_token_id is not None:
            model_config.mask_token_id = config.mask_token_id
        elif getattr(model_config, "mask_token_id", None) is None:
            model_config.mask_token_id = 151669
        if config.init_device == "meta":
            # Keep parameters empty while initializing rotary
            # buffers hold real CPU values for restoration after to_empty().
            with init_empty_weights(), no_init_weights():
                return Qwen3ForBDLM(model_config)
        with torch.device(config.init_device):
            return Qwen3ForBDLM(model_config)

    def parallel_policy(self, model: Any) -> None:
        del model
        return None

    def special_tokens(self, tokenizer: Any, model_config: Any, config: ModelConfig) -> dict[str, int]:
        mask = config.mask_token_id
        if mask is None:
            mask = getattr(model_config, "mask_token_id", 151669)
        eos = config.eos_token_id
        if eos is None:
            eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            eos = getattr(model_config, "eos_token_id", None)
        if eos is None or isinstance(eos, (list, tuple)):
            raise ValueError("Qwen3 order training requires one tokenizer or explicit EOS token ID")
        result = {"mask_token_id": int(mask), "eos_token_id": int(eos)}
        pad = config.pad_token_id
        if pad is None:
            pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(model_config, "pad_token_id", None)
        if pad is not None:
            result["pad_token_id"] = int(pad)
        return result

    @staticmethod
    def _write_loader(target: Path) -> None:
        config_path = target / "config.json"
        exported_config = json.loads(config_path.read_text(encoding="utf-8"))
        exported_config["architectures"] = ["Qwen3ForBDLM"]
        loader_module = "modeling_soldiff" if exported_config.get("p_head_config") else "modeling_dfactory_qwen3"
        # AutoConfig remains Transformers' native Qwen3Config.
        exported_config["auto_map"] = {
            "AutoModel": f"{loader_module}.Qwen3Model",
            "AutoModelForCausalLM": f"{loader_module}.Qwen3ForBDLM",
        }
        if exported_config.get("p_head_config"):
            from dfactory.recipes.soldiff.export import export_soldiff_code

            export_soldiff_code(target, exported_config)
        else:
            (target / f"{loader_module}.py").write_text(
                'from dfactory.models.qwen3 import Qwen3ForBDLM, Qwen3Model\n\n'
                '__all__ = ["Qwen3ForBDLM", "Qwen3Model"]\n',
                encoding="utf-8",
            )
        config_path.write_text(json.dumps(exported_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def save_assets(self, model: Any, tokenizer: Any, output_dir: str | Path) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        model.config.save_pretrained(target)
        tokenizer.save_pretrained(target)
        self._write_loader(target)

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
        self._write_loader(target)


__all__ = ["Qwen3Provider"]
