"""Dataset source metadata."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    factory: Callable[..., Any]
    description: str


__all__ = ["DatasetSpec"]
