"""Trainer lifecycle contracts and shared support types."""

from dfactory.trainers.base import BaseTrainer
from dfactory.trainers.context import StepContext
from dfactory.trainers.metrics import StepMetrics
from dfactory.trainers.state import TrainerState

__all__ = ["BaseTrainer", "StepContext", "StepMetrics", "TrainerState"]
