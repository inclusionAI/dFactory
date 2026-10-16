from contextlib import nullcontext

from dfactory.trainers import StepContext, TrainerState


class FakeLoss:
    def __init__(self, value: float) -> None:
        self.value = value
        self.backward_value = None

    def __mul__(self, scale: float) -> "FakeLoss":
        result = FakeLoss(self.value * scale)
        result.backward_value = self
        return result

    def backward(self) -> None:
        if isinstance(self.backward_value, FakeLoss):
            self.backward_value.backward_value = self.value


class FakeModel:
    def __call__(self, **kwargs):
        return kwargs


def test_context_scales_backward_by_accumulation() -> None:
    loss = FakeLoss(8.0)
    context = StepContext(
        model=FakeModel(),
        state=TrainerState(epoch=2, global_step=3),
        accumulation_steps=4,
        device="cpu",
        forward_context_factory=nullcontext,
        backward_context_factory=nullcontext,
    )
    context.backward(loss, weight=0.5)
    assert loss.backward_value == 1.0
    assert context.epoch == 2
    assert context.global_step == 3
