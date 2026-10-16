"""Fixture specification contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FixtureKind = Literal["file", "directory"]


@dataclass(frozen=True)
class FixtureSpec:
    """Single fixture source-to-target declaration.

    The async executor reads source_uri and sandbox_path to build
    FixtureMapping objects for sandbox_provider.download_fixtures().
    """

    source_path: str
    target_path: str
    kind: FixtureKind = "file"
    required: bool = True
    source_uri: str = ""
    sandbox_path: str = ""


@dataclass(frozen=True)
class FixturePlan:
    """Declarative fixture set for a task.

    Purely declarative — download happens inside the sandbox via
    sandbox_provider.download_fixtures().
    """

    fixtures: list[FixtureSpec] = field(default_factory=list)
