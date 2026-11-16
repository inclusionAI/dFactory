"""Configuration for the fixed block-internal SolDiff method."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class SolDiffConfig:
    block_size: int = 32
    q_config_path: str = ""
    p_head_config_path: str | None = None
    loss_backward_divisor: float = 4096.0
    suffix_reconstruction_tau: float = 8.0
    lambda_align: float = 1.0
    lambda_recon: float = 1.0
    diagnostics_every_steps: int = 100

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> SolDiffConfig:
        unknown = sorted(set(values) - {item.name for item in fields(cls)})
        if unknown:
            raise ValueError(f"unknown soldiff parameter(s): {', '.join(unknown)}")
        try:
            config = cls(**values)
        except TypeError as exc:
            raise ValueError(f"invalid soldiff config: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool) or self.block_size <= 0:
            raise ValueError("recipe.params.block_size must be a positive integer")
        if not isinstance(self.q_config_path, str) or not self.q_config_path.strip():
            raise ValueError("recipe.params.q_config_path must identify a self-contained q configuration")
        if self.p_head_config_path is not None and not isinstance(self.p_head_config_path, str):
            raise ValueError("recipe.params.p_head_config_path must be a path string or null")
        if type(self.diagnostics_every_steps) is not int or self.diagnostics_every_steps <= 0:
            raise ValueError("recipe.params.diagnostics_every_steps must be a positive integer")
        for name in ("loss_backward_divisor", "suffix_reconstruction_tau", "lambda_align", "lambda_recon"):
            value = getattr(self, name)
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"recipe.params.{name} must be a finite number")
            if name in {"loss_backward_divisor", "suffix_reconstruction_tau"} and value <= 0:
                raise ValueError(f"recipe.params.{name} must be positive")


__all__ = ["SolDiffConfig"]
