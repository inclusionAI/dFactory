"""Installable JSON-array dataset source example."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dfactory import DatasetSpec


def build_json_array_source(path: str) -> list[dict[str, Any]]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("json_array sources must contain a JSON array of objects")
    return records


dataset_spec = DatasetSpec(
    name="json_array",
    factory=build_json_array_source,
    description="Records stored in one top-level JSON array.",
)

__all__ = ["build_json_array_source", "dataset_spec"]
