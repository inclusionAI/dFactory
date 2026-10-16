"""Dataset contracts and source discovery."""

from __future__ import annotations

from dfactory.data.batch import AttentionSpec, TrainBatch
from dfactory.data.packing import PackedDataset
from dfactory.data.sources import build_hf_dataset_source, build_jsonl_source
from dfactory.data.spec import DatasetSpec
from dfactory.registry import Registry

_DATASETS: Registry[DatasetSpec] = Registry("dfactory.datasets")


def _register_builtins() -> None:
    if not _DATASETS.contains_builtin("jsonl"):
        _DATASETS.register("jsonl", DatasetSpec("jsonl", build_jsonl_source, "Local JSON Lines records."))
    if not _DATASETS.contains_builtin("hf_dataset"):
        _DATASETS.register(
            "hf_dataset",
            DatasetSpec("hf_dataset", build_hf_dataset_source, "A dataset saved with datasets.save_to_disk."),
        )


def get_dataset(name: str) -> DatasetSpec:
    _register_builtins()
    return _DATASETS.get(name)


def list_datasets() -> tuple[str, ...]:
    _register_builtins()
    return _DATASETS.names()


__all__ = ["AttentionSpec", "DatasetSpec", "PackedDataset", "TrainBatch", "get_dataset", "list_datasets"]
