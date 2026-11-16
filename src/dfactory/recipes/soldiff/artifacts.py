"""Main-head and self-contained q checkpoint assets for the SolDiff recipe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from dfactory.recipes.soldiff.p_head import attach_p_head, get_p_head
from dfactory.recipes.soldiff.q_model import build_q_model


def _owner(model: Any) -> Any:
    current = model
    while current is not None:
        if hasattr(current, "p_head"):
            return current
        current = getattr(current, "module", None)
    raise AttributeError("cannot find p_head on the main model")


def prepare_p_head(
    model: Any,
    *,
    model_path: str = "",
    p_head_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Attach before sharding; actual missing meta parameters initialize in the loader."""
    path = Path(p_head_config_path) if p_head_config_path else None
    if path is None and model_path:
        candidate = Path(model_path) / "p_head_config.json"
        if candidate.is_file():
            path = candidate
    values = dict(getattr(model.config, "p_head_config", {}) or {})
    if path is not None:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("p head config must be a JSON object")
        # Derive fresh-head width and initialization std from the backbone.
        # Retain embedded metadata when restoring an exported head.
        values.update({name: loaded[name] for name in ("intermediate_size", "rms_eps", "activation") if name in loaded})
    config = attach_p_head(model, config_values=values)
    return config.to_dict()


def validate_p_head_resume_config(resolved_config: dict[str, Any], *, checkpoint_dir: str | Path) -> None:
    """Reject p structural changes that tensor-only DCP restore cannot detect."""
    path = Path(checkpoint_dir) / "model_assets" / "p_head_config.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"SolDiff resume requires a readable p head configuration at {path}: {error}") from error
    if not isinstance(saved, dict):
        raise ValueError(f"SolDiff checkpoint p head configuration must be a JSON object: {path}")
    changed = sorted(
        name for name in saved.keys() | resolved_config.keys() if saved.get(name) != resolved_config.get(name)
    )
    if changed:
        raise ValueError(
            f"SolDiff p head configuration differs from the resumed checkpoint: {', '.join(changed)}. "
            f"Rebuild the model with the saved structure in {path}."
        )


def load_p_head_weights(model: Any, *, model_path: str) -> None:
    """Restore an optional separate head after main model materialization."""
    if not model_path:
        return
    path = Path(model_path) / "p_head.safetensors"
    if not path.is_file():
        # Embedded p_head.* tensors are handled with all other main model weights.
        return
    from torch.distributed.tensor import DTensor, distribute_tensor

    head = get_p_head(model)
    parameter = next(head.parameters())
    if parameter.device.type == "meta":
        raise RuntimeError("p head weights must be loaded after the main model is materialized")
    state = load_file(path, device=str(parameter.device))
    target_state = head.state_dict()
    normalized = {}
    for name, tensor in state.items():
        if name.startswith("model.p_head."):
            name = name[len("model.p_head.") :]
        elif name.startswith("p_head."):
            name = name[len("p_head.") :]
        target = target_state.get(name)
        if isinstance(target, DTensor):
            # Each rank reads the same complete sidecar. Slice it to the
            # existing FSDP mesh without broadcast or replacing Parameters.
            tensor = distribute_tensor(
                tensor.to(device=target.device, dtype=target.dtype),
                device_mesh=target.device_mesh,
                placements=target.placements,
                src_data_rank=None,
            )
        normalized[name] = tensor
    head.load_state_dict(normalized, strict=True)


def save_recipe_assets(model: Any, q_model: Any, output_dir: str | Path) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    main = _owner(model)
    config = dict(main.config.p_head_config)
    (target / "p_head_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if q_model is not None:
        _save_q_config(q_model, target / "q_model")


def _save_q_config(q_model: Any, output_dir: str | Path) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    config = getattr(q_model, "_q_package_config", None)
    if config is None:
        raise ValueError("q model has no self-contained package config")
    config = dict(config)
    config["loader"] = "dfactory.recipes.soldiff.artifacts.load_q_model"
    (target / "q_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def save_q_weights(q_model: Any, state_dict: dict[str, Any], output_dir: str | Path) -> None:
    target = _save_q_config(q_model, output_dir)
    # checkpoint_io supplies full ordinary tensors; avoid retaining shared storages.
    tensors = {name: value.detach().cpu().contiguous().clone() for name, value in state_dict.items()}
    save_file(tensors, target / "model.safetensors")


def load_q_model(
    package_dir: str | Path,
    *,
    init_device: str = "cpu",
    torch_dtype: torch.dtype | str = torch.float32,
    attn_implementation: str = "sdpa",
) -> Any:
    """Independently rebuild an exported q package, including all its trained weights."""
    if init_device == "meta":
        raise ValueError("load_q_model requires a materialized device; use build_q_model for meta training setup")
    target = Path(package_dir)
    model = build_q_model(
        target,
        init_device=init_device,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    state = load_file(target / "model.safetensors", device="cpu")
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "prepare_p_head",
    "validate_p_head_resume_config",
    "load_p_head_weights",
    "save_recipe_assets",
    "save_q_weights",
    "load_q_model",
]
