"""Local JSON Lines record source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class JsonlRecordSource(Dataset[dict[str, Any]]):
    """Random-access JSONL source backed by byte offsets."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"JSONL data path is not a file: {self.path}")
        self._offsets: list[int] = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as stream:
            stream.seek(self._offsets[index])
            value = json.loads(stream.readline())
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record {index} must be an object")
        return value


def build_jsonl_source(path: str | Path, **_: Any) -> JsonlRecordSource:
    return JsonlRecordSource(path)


__all__ = ["JsonlRecordSource", "build_jsonl_source"]
