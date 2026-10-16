"""Runtime operations exposed to a trainer's algorithm step."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from dfactory.trainers.state import TrainerState


class StepContext:
    def __init__(
        self,
        *,
        model: Any,
        state: TrainerState,
        accumulation_steps: int,
        device: str,
        forward_context_factory: Callable[[], AbstractContextManager[Any]] = nullcontext,
        backward_context_factory: Callable[[], AbstractContextManager[Any]] = nullcontext,
        reduce_loss: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        if accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        self._model = model
        self._state = state
        self._accumulation_steps = accumulation_steps
        self._device = device
        self._forward_context_factory = forward_context_factory
        self._backward_context_factory = backward_context_factory
        self._reduce_loss = reduce_loss or (lambda loss, _token_count: loss)

    @property
    def global_step(self) -> int:
        return self._state.global_step

    @property
    def epoch(self) -> int:
        return self._state.epoch

    @property
    def accumulation_steps(self) -> int:
        return self._accumulation_steps

    @property
    def device(self) -> str:
        return self._device

    def forward_context(self) -> AbstractContextManager[Any]:
        return self._forward_context_factory()

    def backward_context(self) -> AbstractContextManager[Any]:
        return self._backward_context_factory()

    def forward(self, model_inputs: Mapping[str, Any]) -> Any:
        with self.forward_context():
            return self._model(**dict(model_inputs), use_cache=False, return_dict=True)

    def backward(self, loss: Any, *, weight: float = 1.0) -> None:
        scaled_loss = loss * (weight / self._accumulation_steps)
        with self.backward_context():
            scaled_loss.backward()

    def reduce_loss(self, loss: Any, token_count: Any) -> Any:
        return self._reduce_loss(loss, token_count)


__all__ = ["StepContext"]
