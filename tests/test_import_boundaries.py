import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "dfactory"
PLUGIN_ROOT = Path(__file__).parents[1] / "examples" / "plugins"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
    return result


def test_veomni_imports_stay_inside_gateway_package() -> None:
    violations = []
    for path in SOURCE_ROOT.rglob("*.py"):
        if "runtime/veomni" in path.as_posix():
            continue
        if any(name == "veomni" or name.startswith("veomni.") for name in _imports(path)):
            violations.append(path.relative_to(SOURCE_ROOT).as_posix())
    assert violations == []


def test_recipe_packages_use_common_dfactory_boundaries() -> None:
    violations = []
    for path in (SOURCE_ROOT / "recipes").rglob("*.py"):
        if any(
            name.startswith("dfactory._helpers")
            or name.startswith("dfactory.runtime")
            or name == "veomni"
            or name.startswith("veomni.")
            for name in _imports(path)
        ):
            violations.append(path.relative_to(SOURCE_ROOT).as_posix())
    assert violations == []


def test_helper_package_was_removed() -> None:
    assert not (SOURCE_ROOT / "_helpers").exists()


def test_trainers_only_reach_veomni_through_the_gateway() -> None:
    violations = []
    for path in (SOURCE_ROOT / "trainers").rglob("*.py"):
        imports = _imports(path)
        if any(name == "veomni" or name.startswith("veomni.") for name in imports):
            violations.append(path.relative_to(SOURCE_ROOT).as_posix())
    assert violations == []


def test_no_public_training_framework_package_exists() -> None:
    for package in ("engine", "orchestration", "diffusion", "_training"):
        assert not (SOURCE_ROOT / package).exists()


def test_example_plugins_use_only_public_dfactory_boundaries() -> None:
    violations = []
    for path in PLUGIN_ROOT.rglob("*.py"):
        imports = _imports(path)
        if any(
            name == "veomni" or name.startswith("veomni.") or name.startswith("dfactory._helpers") for name in imports
        ):
            violations.append(path.relative_to(PLUGIN_ROOT).as_posix())
    assert violations == []
