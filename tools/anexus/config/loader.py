"""Config and task payload loading."""

from __future__ import annotations

import json

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class ConfigLoadError(ValueError):
    """Invalid configuration with source location context."""

    def __init__(
        self,
        source_uri: str,
        message: str,
        source_line: int | None = None,
    ) -> None:
        self.source_uri = source_uri
        self.message = message
        self.source_line = source_line
        location = source_uri if source_line is None else f"{source_uri}:{source_line}"
        super().__init__(f"{location}: {message}")


def load_config_data(path: str | Path) -> dict[str, Any]:
    """Load JSON, TOML, or YAML config into a mapping."""

    source_path = Path(path)
    source_uri = source_path.resolve().as_uri()
    return parse_config_text(source_path.read_text(encoding="utf-8"), source_uri=source_uri)


def parse_config_text(text: str, *, source_uri: str) -> dict[str, Any]:
    """Parse JSON, TOML, or YAML config text into a mapping."""

    suffix = Path(urlsplit(source_uri).path).suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(text)
        elif suffix == ".toml":
            data = tomllib.loads(text)
        elif suffix in {".yaml", ".yml"}:
            data = _load_yaml_text(text, source_uri=source_uri)
        else:
            data = _parse_json_without_suffix(text, source_uri=source_uri)
    except JSONDecodeError as exc:
        raise ConfigLoadError(source_uri, f"invalid JSON: {exc.msg}", exc.lineno) from exc
    if not isinstance(data, dict):
        raise ConfigLoadError(source_uri, "config file must contain an object at top level")
    return data


def load_task_payloads(
    path: str | Path,
    *,
    task_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load strict task payloads from JSONL without assigning run metadata."""

    source_path = Path(path)
    source_uri = source_path.resolve().as_uri()
    return parse_task_payloads_text(
        source_path.read_text(encoding="utf-8"),
        source_uri=source_uri,
        task_ids=task_ids,
    )


def parse_task_payloads_text(
    text: str,
    *,
    source_uri: str,
    task_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse strict task payloads from JSONL text."""

    payloads: list[dict[str, Any]] = []
    # JSON strings may legally contain Unicode line/paragraph separators.
    # JSONL records are delimited by LF bytes, so do not use splitlines().
    for source_line, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        payload = _parse_task_payload(line, source_uri=source_uri, source_line=source_line)
        task_id = str(payload["task_id"])
        if task_ids is None or task_id in task_ids:
            payloads.append(payload)
    if not payloads:
        if task_ids:
            requested = ", ".join(sorted(task_ids))
            raise ConfigLoadError(source_uri, f"no matching tasks found for: {requested}")
        raise ConfigLoadError(source_uri, "task source did not contain any tasks")
    return payloads


def _parse_task_payload(
    line: str,
    *,
    source_uri: str,
    source_line: int,
) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except JSONDecodeError as exc:
        raise ConfigLoadError(source_uri, f"invalid JSON: {exc.msg}", source_line) from exc
    if not isinstance(payload, dict):
        raise ConfigLoadError(source_uri, "task payload must be a JSON object", source_line)
    task_id = _string_value(payload.get("task_id"))
    if not task_id:
        task_id = _string_value(payload.get("instance_id"))
        if task_id:
            payload["task_id"] = task_id
    if not task_id:
        raise ConfigLoadError(
            source_uri,
            "task payload requires 'task_id' or 'instance_id'",
            source_line,
        )
    agent_name = _string_value(payload.get("agent_name"))
    if not agent_name:
        raise ConfigLoadError(
            source_uri,
            "task payload requires 'agent_name'",
            source_line,
        )
    return payload


def _string_value(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _load_yaml_text(text: str, *, source_uri: str) -> object:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ConfigLoadError(
            source_uri,
            "YAML config requires PyYAML; use JSON/TOML or install PyYAML",
        ) from exc
    return yaml.safe_load(text)


def _parse_json_without_suffix(text: str, *, source_uri: str) -> object:
    try:
        return json.loads(text)
    except JSONDecodeError as exc:
        raise ConfigLoadError(
            source_uri,
            "unsupported config suffix; expected .json, .toml, .yaml, or .yml",
            exc.lineno,
        ) from exc
