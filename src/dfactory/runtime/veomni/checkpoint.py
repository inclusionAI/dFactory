"""Physical checkpoint I/O delegated to VeOmni."""

from __future__ import annotations

from os import PathLike
from typing import Any

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict


class CheckpointIO:
    def __init__(self, distributed_mode: str) -> None:
        self._implementation = build_checkpointer(ckpt_manager="dcp", dist_backend=distributed_mode)

    def save(self, path: str | PathLike[str], state: dict[str, Any], *, global_step: int) -> None:
        self._implementation.save(str(path), state, save_async=False, global_steps=global_step)

    def load(self, path: str | PathLike[str], state: dict[str, Any]) -> dict[str, Any]:
        return self._implementation.load(str(path), state)

    def model_state_dict(self, path: str | PathLike[str]) -> dict[str, Any]:
        return ckpt_to_state_dict(path, ckpt_manager="dcp")


__all__ = ["CheckpointIO"]
