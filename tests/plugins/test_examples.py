from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "examples" / "plugins"


@pytest.mark.parametrize(
    ("directory", "group", "name", "target"),
    [
        ("custom_recipe", "dfactory.recipes", "example_causal_lm", "dfactory_example_recipe:recipe_spec"),
        ("custom_dataset", "dfactory.datasets", "json_array", "dfactory_example_dataset:dataset_spec"),
        ("custom_model", "dfactory.models", "hf_causal_lm", "dfactory_example_model:model_spec"),
    ],
)
def test_example_declares_loadable_entry_point(
    monkeypatch: pytest.MonkeyPatch,
    directory: str,
    group: str,
    name: str,
    target: str,
) -> None:
    package_root = EXAMPLES / directory
    metadata = tomllib.loads((package_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["entry-points"][group][name] == target

    module_name, attribute = target.split(":", maxsplit=1)
    monkeypatch.syspath_prepend(str(package_root / "src"))
    sys.modules.pop(module_name, None)
    provider = getattr(importlib.import_module(module_name), attribute)
    assert provider.name == name


def test_json_array_example_reads_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package_root = EXAMPLES / "custom_dataset"
    monkeypatch.syspath_prepend(str(package_root / "src"))
    sys.modules.pop("dfactory_example_dataset", None)
    module = importlib.import_module("dfactory_example_dataset")
    source_path = tmp_path / "records.json"
    source_path.write_text('[{"input_ids": [1, 2, 3]}]', encoding="utf-8")

    assert module.dataset_spec.factory(str(source_path)) == [{"input_ids": [1, 2, 3]}]
