"""Workspace dotenv loading."""

from __future__ import annotations

import re
from pathlib import Path

from anexus.config.loader import ConfigLoadError

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_env_file(path: str | Path) -> dict[str, str]:
    """Load a small dotenv-compatible KEY=VALUE file."""

    env_path = Path(path).expanduser()
    source_uri = env_path.resolve().as_uri()
    if not env_path.exists():
        return {}
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigLoadError(source_uri, f"could not read env file: {exc}") from exc

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not _ENV_KEY_RE.match(key):
            raise ConfigLoadError(source_uri, "env line must use KEY=VALUE", line_number)
        values[key] = _strip_env_quotes(value.strip())
    return values


def _strip_env_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
