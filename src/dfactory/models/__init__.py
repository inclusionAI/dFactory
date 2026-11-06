"""Built-in and external model-provider discovery."""

from __future__ import annotations

from dfactory.models.base import ModelProvider
from dfactory.models.spec import ModelSpec
from dfactory.registry import Registry

_MODELS: Registry[ModelSpec] = Registry("dfactory.models")


def _register_builtins() -> None:
    for name, module in (("llada2", "dfactory.models.llada2"), ("qwen3", "dfactory.models.qwen3")):
        if not _MODELS.contains_builtin(name):
            _MODELS.register_lazy(name, module, "model_spec")


def get_model(name: str) -> ModelSpec:
    _register_builtins()
    return _MODELS.get(name)


def list_models() -> tuple[str, ...]:
    _register_builtins()
    return _MODELS.names()


__all__ = ["ModelProvider", "ModelSpec", "get_model", "list_models"]
