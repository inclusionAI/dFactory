"""Stable ID helpers."""

from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str) -> str:
    """Return a compact, prefixed random id."""

    cleaned = prefix.strip().lower().replace("_", "-")
    if not cleaned:
        cleaned = "id"
    return f"{cleaned}_{uuid4().hex[:16]}"
