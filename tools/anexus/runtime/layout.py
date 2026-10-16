"""Run directory layout and stable identifier helpers."""

from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_SLUG_STRIP = re.compile(r"[^a-z0-9._-]+")
_SLUG_DASH_RUN = re.compile(r"-{2,}")


def slug(value: str, *, max_len: int = 64, fallback: str = "unknown") -> str:
    """Return a lowercase filesystem-safe slug."""

    cleaned = _SLUG_STRIP.sub("-", value.lower())
    cleaned = _SLUG_DASH_RUN.sub("-", cleaned).strip("._-")
    cleaned = cleaned[:max_len].rstrip("._-")
    return cleaned or fallback


def make_run_id(
    agent_name: str,
    fingerprint: str,
    *,
    now: datetime | None = None,
) -> str:
    """Return ``<timestamp>-<agent_slug>-<fingerprint8>`` run id."""

    timestamp = now or datetime.now(UTC)
    timestamp_text = timestamp.strftime("%Y%m%d-%H%M")
    agent_slug = slug(agent_name, max_len=20, fallback="agent")
    fingerprint_prefix = fingerprint[:8] if fingerprint else "00000000"
    return f"{timestamp_text}-{agent_slug}-{fingerprint_prefix}"


@dataclass(frozen=True)
class RunLayout:
    """Path helper for one run directory under a run root."""

    base_dir: Path
    run_id: str

    @classmethod
    def create(
        cls,
        workspace: Path,
        *,
        agent_name: str,
        fingerprint: str,
        now: datetime | None = None,
        runs_dir_name: str = "runs",
    ) -> RunLayout:
        base_dir = workspace / runs_dir_name
        run_id = make_run_id(agent_name, fingerprint, now=now)
        for index in range(2, 100):
            if not (base_dir / run_id).exists():
                break
            run_id = f"{make_run_id(agent_name, fingerprint, now=now)}-n{index:02d}"
        return cls(base_dir=base_dir, run_id=run_id)

    @classmethod
    def open_existing(cls, run_dir: Path) -> RunLayout:
        return cls(base_dir=run_dir.parent, run_id=run_dir.name)

    @property
    def run_dir(self) -> Path:
        return self.base_dir / self.run_id

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def run_lock_path(self) -> Path:
        return self.run_dir / "run_lock.json"

    @property
    def latest_path(self) -> Path:
        return self.base_dir / "latest"

    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def update_latest_link(self) -> None:
        """Point ``runs/latest`` at this run, falling back to a text file."""

        self.base_dir.mkdir(parents=True, exist_ok=True)
        temporary_link = self.latest_path.with_name(f".latest.{os.getpid()}.tmp")
        with contextlib.suppress(OSError):
            temporary_link.unlink()
        try:
            temporary_link.symlink_to(self.run_id)
            temporary_link.replace(self.latest_path)
        except OSError:
            with contextlib.suppress(OSError):
                self.latest_path.with_suffix(".txt").write_text(self.run_id, encoding="utf-8")
