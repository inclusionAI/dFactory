import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[1]


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"dfactory_{name}", ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


download_hf_model = _load_script("download_hf_model")
moe_convertor = _load_script("moe_convertor")


def test_download_uses_the_exact_requested_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call: dict[str, str] = {}
    monkeypatch.setattr(download_hf_model, "snapshot_download", lambda **kwargs: call.update(kwargs))

    download_hf_model.main("inclusionAI/LLaDA2.0-mini-preview", str(tmp_path))

    assert call == {
        "repo_id": "inclusionAI/LLaDA2.0-mini-preview",
        "local_dir": str(tmp_path),
    }


def test_moe_converter_rejects_an_unknown_mode_before_loading_assets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode must be merge or split"):
        moe_convertor.main(str(tmp_path / "input"), str(tmp_path / "output"), "unknown")
