"""Registry for host-side task graders."""

from __future__ import annotations

from anexus.tasks.graders.base import BaseGrader

_GRADERS: dict[str, type[BaseGrader]] = {}


def register_grader(grader_cls: type[BaseGrader]) -> type[BaseGrader]:
    """Register a grader class by its `name` attribute."""

    if not issubclass(grader_cls, BaseGrader):
        raise TypeError(f"grader class {grader_cls.__name__} must subclass BaseGrader")
    name = getattr(grader_cls, "name", "").strip()
    if not name:
        raise ValueError(f"grader class {grader_cls.__name__} must define a non-empty name")
    _GRADERS[name] = grader_cls
    return grader_cls


def get_grader(name: str) -> BaseGrader:
    try:
        return _GRADERS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown grader: {name!r}. Available: {available_graders()}") from exc


def available_graders() -> list[str]:
    return sorted(_GRADERS)
