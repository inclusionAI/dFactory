"""Hugging Face dataset source."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_hf_dataset_source(path: str | Path, **_: Any) -> Any:
    from datasets import load_from_disk

    return load_from_disk(str(path))


__all__ = ["build_hf_dataset_source"]
