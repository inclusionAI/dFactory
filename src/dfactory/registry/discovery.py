"""Lazy discovery for built-in and third-party dFactory providers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from typing import Generic, TypeVar

from dfactory.registry.errors import DuplicateProviderError, ProviderNotFoundError

_T = TypeVar("_T")


@dataclass(frozen=True)
class _LazyProvider(Generic[_T]):
    module: str
    attribute: str

    def load(self) -> _T:
        return getattr(import_module(self.module), self.attribute)


class Registry(Generic[_T]):
    """A small registry whose external providers are loaded only when selected."""

    def __init__(self, entry_point_group: str) -> None:
        self.entry_point_group = entry_point_group
        self._builtins: dict[str, _T | _LazyProvider[_T]] = {}

    def register(self, name: str, provider: _T) -> None:
        if not name:
            raise ValueError("provider name must not be empty")
        if name in self._builtins or name in self._external():
            raise DuplicateProviderError(f"provider {name!r} is already registered")
        self._builtins[name] = provider

    def register_lazy(self, name: str, module: str, attribute: str) -> None:
        if not name:
            raise ValueError("provider name must not be empty")
        if name in self._builtins or name in self._external():
            raise DuplicateProviderError(f"provider {name!r} is already registered")
        self._builtins[name] = _LazyProvider(module, attribute)

    def names(self) -> tuple[str, ...]:
        external = self._external()
        duplicates = set(self._builtins).intersection(external)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise DuplicateProviderError(f"built-in and external providers use the same name(s): {names}")
        return tuple(sorted((*self._builtins, *external)))

    def contains(self, name: str) -> bool:
        return name in self._builtins or name in self._external()

    def contains_builtin(self, name: str) -> bool:
        return name in self._builtins

    def get(self, name: str) -> _T:
        external = self._external()
        if name in self._builtins and name in external:
            raise DuplicateProviderError(f"built-in and external providers both use {name!r}")
        if name in self._builtins:
            provider = self._builtins[name]
            return provider.load() if isinstance(provider, _LazyProvider) else provider
        if name in external:
            return external[name].load()
        available = ", ".join(self.names()) or "none"
        raise ProviderNotFoundError(f"provider {name!r} was not found; available: {available}")

    def _external(self) -> dict[str, EntryPoint]:
        selected = entry_points().select(group=self.entry_point_group)
        result: dict[str, EntryPoint] = {}
        for entry_point in selected:
            if entry_point.name in result:
                raise DuplicateProviderError(
                    f"multiple entry points named {entry_point.name!r} in {self.entry_point_group!r}"
                )
            result[entry_point.name] = entry_point
        return result


__all__ = ["Registry"]
