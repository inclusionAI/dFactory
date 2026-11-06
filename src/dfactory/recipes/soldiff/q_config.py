"""Read q architecture metadata without loading a model or distributed runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_q_config(path: str | Path) -> dict[str, Any]:
    """Read complete architecture metadata without reading teacher or q weights."""
    config_path = Path(path)
    if config_path.is_dir():
        preferred = config_path / "q_config.json"
        config_path = preferred if preferred.is_file() else config_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read q config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("q config must be a JSON object")
    layers = config.get("encoder_num_layers")
    if type(layers) is not int or layers <= 0:
        raise ValueError("q config requires a positive integer encoder_num_layers")
    if config.get("q_head_type", "linear") != "linear":
        raise ValueError("q config q_head_type must be linear")
    if config.get("base_model_type", "qwen3") != "qwen3":
        raise ValueError("q config base_model_type must be qwen3")
    if "rms_eps" not in config or float(config["rms_eps"]) <= 0:
        raise ValueError("q config requires positive rms_eps")
    base_config = config.get("base_config")
    if not isinstance(base_config, dict):
        teacher_path = config.get("teacher_config_path")
        if not teacher_path:
            raise ValueError("q config requires inline base_config or a local teacher_config_path")
        teacher_path = Path(teacher_path)
        if not teacher_path.is_absolute():
            teacher_path = config_path.parent / teacher_path
        if teacher_path.is_dir():
            teacher_path = teacher_path / "config.json"
        try:
            base_config = json.loads(teacher_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read local q teacher config {teacher_path}: {error}") from error
    if not isinstance(base_config, dict) or base_config.get("model_type", "qwen3") != "qwen3":
        raise ValueError("q base_config must be a native Qwen3 configuration")
    # Reject incomplete architecture metadata instead of silently using Qwen3 defaults.
    required = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
    )
    missing = [name for name in required if name not in base_config]
    if missing:
        raise ValueError(f"q base_config is missing architecture field(s): {', '.join(missing)}")
    if layers > int(base_config["num_hidden_layers"]):
        raise ValueError("q encoder_num_layers exceeds base_config.num_hidden_layers")
    if "hidden_size" in config and int(config["hidden_size"]) != int(base_config["hidden_size"]):
        raise ValueError("q hidden_size does not match base_config.hidden_size")
    base_config = dict(base_config)
    base_config["num_hidden_layers"] = layers
    if isinstance(base_config.get("layer_types"), list):
        base_config["layer_types"] = base_config["layer_types"][:layers]
    result = dict(config)
    result["base_config"] = base_config
    result["base_model_type"] = "qwen3"
    result["encoder_num_layers"] = layers
    result["rms_eps"] = float(config["rms_eps"])
    result["q_score_bias"] = bool(config.get("q_score_bias", False))
    result["q_head_type"] = "linear"
    result["contains_lm_head"] = False
    result["tie_word_embeddings"] = False
    # Embedded base_config takes precedence over external configuration references.
    # Treat auto_map as metadata rather than a loader directive.
    if "auto_map" in result:
        result["source_auto_map"] = result.pop("auto_map")
    return result


__all__ = ["load_q_config"]
