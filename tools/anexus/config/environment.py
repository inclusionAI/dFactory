"""Workspace environment-reference resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping

from dotenv.variables import Atom, Literal, Variable, parse_variables

from anexus.config.loader import ConfigLoadError

_ENV_NAMESPACE_PREFIX = "${env:"
_POSIX_VARIABLE_PREFIX = "${"


def resolve_env_refs(
    value: object,
    *,
    env: Mapping[str, str] | None = None,
    source_uri: str = "memory://config",
) -> object:
    """Resolve `${env:NAME}` and `${env:NAME:-fallback}` references."""
    source_env = os.environ if env is None else env
    return _resolve_env_refs(value, env=source_env, source_uri=source_uri, path="")


def env_refs_resolvable(value: object, *, env: Mapping[str, str]) -> bool:
    """Return whether every environment reference has a value or fallback."""
    if not isinstance(value, str) or _ENV_NAMESPACE_PREFIX not in value:
        return True
    return all(atom.name in env or atom.default is not None for atom in _variables(value))


def _resolve_env_refs(
    value: object,
    *,
    env: Mapping[str, str],
    source_uri: str,
    path: str,
) -> object:
    if isinstance(value, dict):
        return {
            key: _resolve_env_refs(
                item,
                env=env,
                source_uri=source_uri,
                path=f"{path}.{key}" if path else str(key),
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_env_refs(
                item,
                env=env,
                source_uri=source_uri,
                path=f"{path}[{index}]" if path else f"[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    if _ENV_NAMESPACE_PREFIX not in value:
        return value

    atoms = _parse_env_atoms(value, source_uri=source_uri, path=path)
    for variable in (atom for atom in atoms if isinstance(atom, Variable)):
        if variable.name not in env and variable.default is None:
            location = path or "<root>"
            raise ConfigLoadError(
                source_uri,
                f"{location}: missing required environment variable {variable.name}",
            )
    return "".join(atom.resolve(env) for atom in atoms)


def _variables(value: str) -> tuple[Variable, ...]:
    return tuple(atom for atom in parse_variables(_normalize_env_refs(value)) if isinstance(atom, Variable))


def _parse_env_atoms(value: str, *, source_uri: str, path: str) -> tuple[Atom, ...]:
    atoms = tuple(parse_variables(_normalize_env_refs(value)))
    if any(isinstance(atom, Literal) and _POSIX_VARIABLE_PREFIX in atom.value for atom in atoms):
        location = path or "<root>"
        raise ConfigLoadError(
            source_uri,
            f"{location}: invalid environment reference: {value}",
        )
    return atoms


def _normalize_env_refs(value: str) -> str:
    """Adapt ANexus's env namespace to python-dotenv's POSIX interpolation."""
    return value.replace(_ENV_NAMESPACE_PREFIX, _POSIX_VARIABLE_PREFIX)
