"""URI helpers for agent catalogs and task sources."""

from __future__ import annotations

import posixpath
from urllib.parse import urlsplit, urlunsplit


def base_uri(uri: str) -> str:
    if not uri:
        return ""
    parts = urlsplit(uri)
    path = parts.path.rsplit("/", 1)[0]
    return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")


def resolve_uri(catalog_uri: str, value: str) -> str:
    if not value:
        return ""
    value_parts = urlsplit(value)
    if value_parts.scheme:
        return value
    base = base_uri(catalog_uri)
    if not base:
        return value
    base_parts = urlsplit(base)
    relative = value.removeprefix("./")
    path = posixpath.normpath(posixpath.join(base_parts.path, relative))
    if base_parts.path.startswith("/") and not path.startswith("/"):
        path = f"/{path}"
    return urlunsplit((base_parts.scheme, base_parts.netloc, path, "", ""))
