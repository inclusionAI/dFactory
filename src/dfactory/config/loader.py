"""Strict YAML loading and dotted overrides for dFactory configs."""

from __future__ import annotations

from dataclasses import fields
from os import PathLike
from pathlib import Path
from typing import Any, TypeVar

import yaml

from dfactory.config.schema import (
    CheckpointConfig,
    DataConfig,
    DistributedConfig,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
    RecipeConfig,
    RunConfig,
    RunSettings,
)
from dfactory.config.validation import validate_config


class ConfigError(ValueError):
    """Raised when a config cannot be parsed into the public schema."""


_T = TypeVar("_T")
_SECTIONS = {
    "run": RunSettings,
    "recipe": RecipeConfig,
    "model": ModelConfig,
    "data": DataConfig,
    "distributed": DistributedConfig,
    "optimization": OptimizationConfig,
    "checkpoint": CheckpointConfig,
    "logging": LoggingConfig,
}


def _construct(config_type: type[_T], values: Any, path: str) -> _T:
    if not isinstance(values, dict):
        raise ConfigError(f"{path} must be a mapping")
    allowed = {item.name for item in fields(config_type)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"unknown field(s) in {path}: {', '.join(unknown)}")
    try:
        return config_type(**values)
    except TypeError as exc:
        raise ConfigError(f"invalid {path}: {exc}") from exc


def _apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"override must use PATH=VALUE: {override}")
    dotted_path, raw_value = override.split("=", 1)
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        raise ConfigError(f"override path must not be empty: {override}")

    if len(parts) < 2 or parts[0] not in _SECTIONS:
        raise ConfigError(f"unknown override path: {dotted_path}")
    section_name, field_name = parts[:2]
    allowed_fields = {item.name for item in fields(_SECTIONS[section_name])}
    if field_name not in allowed_fields:
        raise ConfigError(f"unknown override path: {dotted_path}")

    section = data.setdefault(section_name, {})
    if not isinstance(section, dict):
        raise ConfigError(f"override path is not a mapping: {section_name}")
    if len(parts) == 2:
        section[field_name] = yaml.safe_load(raw_value)
        return

    if (section_name, field_name) != ("recipe", "params"):
        raise ConfigError(f"unknown override path: {dotted_path}")
    current = section.setdefault("params", {})
    if not isinstance(current, dict):
        raise ConfigError("override path is not a mapping: recipe.params")
    for index, part in enumerate(parts[2:-1], start=2):
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"override path is not a mapping: {'.'.join(parts[: index + 1])}")
        current = child
    current[parts[-1]] = yaml.safe_load(raw_value)


def load_config(path: str | PathLike[str], overrides: tuple[str, ...] | list[str] = ()) -> RunConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    unknown_sections = sorted(set(raw) - set(_SECTIONS))
    if unknown_sections:
        raise ConfigError(f"unknown config section(s): {', '.join(unknown_sections)}")

    for override in overrides:
        _apply_override(raw, override)

    sections = {name: _construct(section_type, raw.get(name, {}), name) for name, section_type in _SECTIONS.items()}
    config = RunConfig(**sections)
    try:
        validate_config(config)
        from dfactory.data import get_dataset
        from dfactory.models import get_model
        from dfactory.recipes import get_recipe

        get_dataset(config.data.source)
        get_model(config.model.provider)
        recipe_spec = get_recipe(config.recipe.name)
        recipe_config = recipe_spec.config_type.from_mapping(config.recipe.params)
        recipe_spec.validate_run(recipe_config=recipe_config, run_config=config)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    except KeyError as exc:
        raise ConfigError(str(exc)) from exc
    return config


__all__ = ["ConfigError", "load_config"]
