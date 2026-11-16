"""Block generation with learned SolDiff position scores."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn


@dataclass(frozen=True)
class SolDiffGenerationConfig:
    strategy: Literal["prior_static", "prior_confidence_dynamic"] = "prior_static"
    max_new_tokens: int = 1024
    block_length: int = 32
    steps_per_block: int = 32
    prior_temperature: float = 0.0
    confidence_threshold: float = 0.9
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    use_cache: bool = True
    eos_early_stop: bool = False
    eos_token_id: int | tuple[int, ...] | list[int] | None = (151643, 151645)
    mask_token_id: int = 151669

    @property
    def eos_ids(self) -> tuple[int, ...]:
        value = self.eos_token_id
        if value is None:
            return ()
        values = (value,) if type(value) is int else value
        if not isinstance(values, (tuple, list)) or any(type(item) is not int or item < 0 for item in values):
            raise ValueError("eos_token_id must be a non-negative integer, a list of IDs, or None")
        return tuple(dict.fromkeys(values))

    def validate(self) -> None:
        if self.strategy not in {"prior_static", "prior_confidence_dynamic"}:
            raise ValueError("strategy must be prior_static or prior_confidence_dynamic")
        for name in ("block_length", "steps_per_block"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if type(self.mask_token_id) is not int or self.mask_token_id < 0:
            raise ValueError("mask_token_id must be a non-negative integer")
        for name in ("temperature", "prior_temperature", "confidence_threshold"):
            value = getattr(self, name)
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.confidence_threshold > 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 0):
            raise ValueError("top_k must be a non-negative integer or None")
        if self.top_p is not None and (
            not isinstance(self.top_p, Real) or isinstance(self.top_p, bool) or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be in (0, 1] or None")
        if type(self.use_cache) is not bool or type(self.eos_early_stop) is not bool:
            raise ValueError("use_cache and eos_early_stop must be booleans")
        if self.mask_token_id in self.eos_ids:
            raise ValueError("mask_token_id must differ from every EOS ID")


def _head_config(values: Any) -> dict[str, Any]:
    from dfactory.models.qwen3.heads import PWarmupHeadConfig

    if not isinstance(values, dict):
        raise ValueError("p_head_config must be a JSON object")
    try:
        config = PWarmupHeadConfig(**values)
        return config.to_dict()
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid p_head_config: {error}") from error


def load_soldiff_model(
    model_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: Literal["eager", "sdpa", "flex_attention"] = "sdpa",
) -> nn.Module:
    """Load a local HF model with its trained p head and saved architecture."""
    from transformers import Qwen3Config

    from dfactory.models.qwen3 import Qwen3ForBDLM

    path = Path(model_path)
    raw_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if raw_config.get("model_type") != "qwen3":
        raise ValueError("SolDiff generation requires a Qwen3 checkpoint")
    embedded = raw_config.get("p_head_config")
    separate_path = path / "p_head_config.json"
    separate = json.loads(separate_path.read_text(encoding="utf-8")) if separate_path.is_file() else None
    if embedded is None and separate is None:
        raise ValueError("checkpoint requires config.p_head_config or p_head_config.json")
    resolved = _head_config(embedded if embedded is not None else separate)
    if separate is not None and _head_config(separate) != resolved:
        raise ValueError("p_head_config.json conflicts with config.p_head_config")
    config = Qwen3Config.from_dict(raw_config)
    config.p_head_config = resolved
    model, loading_info = Qwen3ForBDLM.from_pretrained(
        path,
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        output_loading_info=True,
        key_mapping={r"^model\.p_head\.": "p_head."},
    )
    missing = list(loading_info["missing_keys"])
    head_missing = [name for name in missing if name.startswith("p_head.")]
    weights_path = path / "p_head.safetensors"
    if len(head_missing) == len(model.p_head.state_dict()) and weights_path.is_file():
        from safetensors.torch import load_file

        state = load_file(weights_path, device="cpu")
        state = {name.removeprefix("model.").removeprefix("p_head."): tensor for name, tensor in state.items()}
        model.p_head.load_state_dict(state, strict=True)
        missing = [name for name in missing if name not in head_missing]
    if missing or loading_info["unexpected_keys"]:
        raise ValueError(
            f"checkpoint parameters do not match SolDiff: missing={missing}, "
            f"unexpected={loading_info['unexpected_keys']}"
        )
    return model.to(device).eval()


def _predict_tokens(logits: torch.Tensor, config: SolDiffGenerationConfig) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float()
    if config.temperature > 0 and config.temperature != 1:
        logits = logits / config.temperature
    if config.top_k:
        if config.top_k > logits.shape[-1]:
            raise ValueError("top_k exceeds the model vocabulary size")
        cutoff = torch.topk(logits, config.top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, -torch.inf)
    if config.top_p is not None and config.top_p < 1:
        sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
        remove = sorted_logits.softmax(-1).cumsum(-1) > config.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits = logits.masked_fill(torch.zeros_like(remove).scatter(-1, sorted_indices, remove), -torch.inf)
    probabilities = logits.softmax(-1)
    tokens = probabilities.argmax(-1)
    confidence = probabilities.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, confidence


def _select_positions(
    scores: torch.Tensor,
    confidence: torch.Tensor,
    active: torch.Tensor,
    count: int,
    config: SolDiffGenerationConfig,
    generator: torch.Generator | None,
) -> torch.Tensor:
    selected = torch.zeros_like(active)
    available = int(active.sum().item())
    count = min(count, available)
    if not available or (config.strategy == "prior_static" and count == 0):
        return selected
    scores = scores.float().masked_fill(~active, -torch.inf)[0]
    order_length = count if config.strategy == "prior_static" else available
    if config.prior_temperature == 0:
        order = torch.topk(scores, k=order_length).indices
    else:
        probabilities = (scores / config.prior_temperature).softmax(-1)
        if torch.isfinite(probabilities).all() and int((probabilities > 0).sum()) >= order_length:
            order = torch.multinomial(probabilities, order_length, replacement=False, generator=generator)
        else:
            # Re-normalize after each draw when low temperatures underflow.
            remaining = active[0].nonzero(as_tuple=True)[0]
            draws = []
            for _ in range(order_length):
                remaining_scores = scores[remaining]
                probabilities = ((remaining_scores - remaining_scores.max()) / config.prior_temperature).softmax(-1)
                draw = torch.multinomial(probabilities, 1, generator=generator)
                draws.append(remaining[draw])
                remaining = remaining[torch.arange(remaining.numel(), device=scores.device) != draw]
            order = torch.cat(draws)
    if config.strategy == "prior_confidence_dynamic":
        failures = (confidence[0, order] <= config.confidence_threshold).nonzero(as_tuple=True)[0]
        prefix_length = int(failures[0].item()) if failures.numel() else order.numel()
        count = max(count, prefix_length)
    selected[0, order[:count]] = True
    return selected


def _attention_mask(model: nn.Module, start: int, length: int, keys: int, block: int, device: torch.device) -> Any:
    if model.config._attn_implementation == "flex_attention":
        from torch.nn.attention.flex_attention import create_block_mask

        def block_mask(batch, head, query, key):
            return (query < length) & (key < keys) & (key // block <= (start + query) // block)

        return create_block_mask(block_mask, B=1, H=None, Q_LEN=length, KV_LEN=keys, device=str(device))
    query_blocks = torch.arange(start, start + length, device=device) // block
    key_blocks = torch.arange(keys, device=device) // block
    allowed = key_blocks.unsqueeze(0) <= query_blocks.unsqueeze(1)
    dtype = getattr(model, "dtype", torch.float32)
    return torch.zeros((1, 1, length, keys), dtype=dtype, device=device).masked_fill(~allowed, -torch.inf)


def _cache_length(cache: Any) -> int:
    return int(cache.get_seq_length()) if cache is not None else 0


def _check_cache(cache: Any, expected: int) -> None:
    actual = _cache_length(cache)
    if actual != expected:
        raise RuntimeError(f"prefix cache length is {actual}; expected {expected}")


def _fork_cache(cache: Any) -> Any:
    if cache is None:
        return None
    if hasattr(cache, "layers"):
        fork = copy.copy(cache)
        fork.layers = [copy.copy(layer) for layer in cache.layers]
        return fork
    if hasattr(cache, "to_legacy_cache") and hasattr(type(cache), "from_legacy_cache"):
        return type(cache).from_legacy_cache(cache.to_legacy_cache())
    raise TypeError(f"unsupported prefix cache type: {type(cache).__name__}")


def _first_eos(tokens: torch.Tensor, eos_ids: tuple[int, ...]) -> int | None:
    matches = torch.zeros_like(tokens, dtype=torch.bool)
    for token_id in eos_ids:
        matches |= tokens == token_id
    positions = matches.nonzero(as_tuple=True)[0]
    return int(positions[0].item()) if positions.numel() else None


@torch.inference_mode()
def generate_soldiff(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: SolDiffGenerationConfig | None = None,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate one unpadded prompt; return prompt plus completion, including EOS.

    Token choices are greedy. ``prior_temperature`` controls position sampling.
    Completed blocks are cached; a temporary cache is used for each denoising
    forward. The final working block is decoded in full before length cropping.
    """
    from dfactory.models.qwen3.heads import get_p_head

    config = config or SolDiffGenerationConfig()
    config.validate()
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be an unpadded torch.long tensor shaped [1, prompt_length]")
    if (input_ids == config.mask_token_id).any().item():
        raise ValueError("the prompt must not contain the mask token")
    if config.max_new_tokens == 0:
        return input_ids.clone()
    model.eval()
    model.config.return_dict = True
    head = get_p_head(model)
    prompt_length = input_ids.shape[1]
    target_length = prompt_length + config.max_new_tokens
    block = config.block_length
    total_length = ((target_length + block - 1) // block) * block
    position_ids = torch.arange(total_length, device=input_ids.device).unsqueeze(0)
    tokens = torch.full((1, total_length), config.mask_token_id, dtype=torch.long, device=input_ids.device)
    tokens[:, :prompt_length] = input_ids
    prefill_length = (prompt_length // block) * block
    steps = min(config.steps_per_block, config.max_new_tokens)
    schedule = [block // steps + (index < block % steps) for index in range(steps)]
    prefix_cache = None
    if config.use_cache and prefill_length:
        output = model(
            input_ids=tokens[:, :prefill_length],
            attention_mask=_attention_mask(model, 0, prefill_length, prefill_length, block, input_ids.device),
            position_ids=position_ids[:, :prefill_length],
            cache_position=position_ids[0, :prefill_length],
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        prefix_cache = output.past_key_values
        del output
        _check_cache(prefix_cache, prefill_length)

    for start in range(prefill_length, total_length, block):
        end = start + block
        if config.use_cache:
            _check_cache(prefix_cache, start)
        block_tokens = tokens[:, start:end].clone()
        query_start = start if config.use_cache else 0
        attention_mask = _attention_mask(model, query_start, end - query_start, end, block, input_ids.device)
        for count in schedule:
            active = block_tokens == config.mask_token_id
            if not active.any().item():
                break
            query_tokens = block_tokens if config.use_cache else tokens[:, :end]
            step_cache = _fork_cache(prefix_cache) if config.use_cache else None
            output = model(
                input_ids=query_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids[:, query_start:end],
                cache_position=position_ids[0, query_start:end],
                past_key_values=step_cache,
                output_hidden_states=True,
                use_cache=step_cache is not None,
                return_dict=True,
            )
            if step_cache is not None:
                _check_cache(step_cache, end)
                _check_cache(prefix_cache, start)
            scores = head(output.hidden_states[-1])[:, -block:].float()
            candidates, confidence = _predict_tokens(output.logits[:, -block:], config)
            selected = _select_positions(scores, confidence, active, count, config, generator)
            del output
            block_tokens[selected] = candidates[selected]
            tokens[:, start:end] = block_tokens
            if config.eos_early_stop:
                eos = _first_eos(tokens[0, prompt_length:target_length], config.eos_ids)
                if eos is not None and (tokens[0, prompt_length : prompt_length + eos] != config.mask_token_id).all():
                    return tokens[:, : prompt_length + eos + 1]

        if _first_eos(tokens[0, prompt_length : min(end, target_length)], config.eos_ids) is not None:
            break
        if config.use_cache and end < total_length:
            output = model(
                input_ids=block_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids[:, start:end],
                cache_position=position_ids[0, start:end],
                past_key_values=prefix_cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            prefix_cache = output.past_key_values
            del output
            _check_cache(prefix_cache, end)

    result = tokens[:, :target_length]
    eos = _first_eos(result[0, prompt_length:], config.eos_ids)
    return result if eos is None else result[:, : prompt_length + eos + 1]


__all__ = ["SolDiffGenerationConfig", "generate_soldiff", "load_soldiff_model"]
