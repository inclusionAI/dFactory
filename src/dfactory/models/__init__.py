"""Built-in and external model-provider discovery."""

from __future__ import annotations

from dfactory.models.base import ModelProvider
from dfactory.models.spec import ModelSpec
from dfactory.registry import Registry

_MODELS: Registry[ModelSpec] = Registry("dfactory.models")


def _register_builtins() -> None:
    if _MODELS.contains_builtin("llada2"):
        return
    _MODELS.register_lazy("llada2", "dfactory.models.llada2", "model_spec")


def get_model(name: str) -> ModelSpec:
    _register_builtins()
    return _MODELS.get(name)


def list_models() -> tuple[str, ...]:
    _register_builtins()
    return _MODELS.names()


__all__ = ["ModelProvider", "ModelSpec", "get_model", "list_models"]
