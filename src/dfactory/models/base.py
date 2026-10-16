"""Public model-provider contract."""

from __future__ import annotations

from os import PathLike
from typing import Any, Protocol

from dfactory.config import ModelConfig


class ModelProvider(Protocol):
    def load_tokenizer(self, config: ModelConfig) -> Any: ...

    def build_model(self, config: ModelConfig) -> Any: ...

    def parallel_policy(self, model: Any) -> Any: ...

    def special_tokens(self, tokenizer: Any, model_config: Any, config: ModelConfig) -> dict[str, int]: ...

    def save_assets(self, model: Any, tokenizer: Any, output_dir: str | PathLike[str]) -> None: ...

    def save_weights(
        self,
        model: Any,
        tokenizer: Any,
        state_dict: dict[str, Any],
        output_dir: str | PathLike[str],
    ) -> None: ...


__all__ = ["ModelProvider"]
