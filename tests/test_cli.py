import pytest

from dfactory.cli import main


def test_top_level_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "Train and extend discrete diffusion language models" in capsys.readouterr().out


def test_recipe_list_contains_builtin_recipes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["recipes", "list"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "ar2bd",
        "ar_sft",
        "emtf",
        "emtf_random_t2t",
        "estf",
        "mtf",
        "sft",
    ]


def test_dataset_list_contains_builtins(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["datasets", "list"]) == 0
    assert capsys.readouterr().out.splitlines() == ["hf_dataset", "jsonl"]


def test_model_list_contains_llada2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models", "list"]) == 0
    assert capsys.readouterr().out.splitlines() == ["llada2"]
