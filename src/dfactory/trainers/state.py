"""Mutable progress state owned by a trainer."""

from dataclasses import dataclass


@dataclass
class TrainerState:
    epoch: int = 0
    step_in_epoch: int = 0
    global_step: int = 0

    def advance(self) -> None:
        self.step_in_epoch += 1
        self.global_step += 1

    def next_epoch(self) -> None:
        self.epoch += 1
        self.step_in_epoch = 0


__all__ = ["TrainerState"]
