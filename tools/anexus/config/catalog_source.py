"""Git-backed catalog source synchronization."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath

from anexus.config.loader import ConfigLoadError

_COMMIT_ID_RE = re.compile(r"[0-9a-fA-F]{7,64}")


@dataclass(frozen=True)
class CatalogSyncResult:
    """Content and provenance for one synchronized catalog file."""

    repository: str
    branch: str
    path: str
    commit_id: str
    resolved_commit_id: str
    source_uri: str
    content: str
    fetched_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class CatalogSource:
    """Select exactly one catalog file from a Git branch or commit."""

    repository: str
    branch: str
    path: str
    commit_id: str = ""

    @classmethod
    def from_workspace_config(cls, config: Mapping[str, object]) -> CatalogSource:
        """Build a catalog locator from an ``anexus.config.yaml`` mapping."""
        configured = config.get("catalog")
        catalog = configured if isinstance(configured, Mapping) else {}
        return cls(
            repository=str(catalog.get("repository", "") or "").strip(),
            branch=str(catalog.get("branch", "") or "").strip(),
            path=str(catalog.get("path", "") or "").strip(),
            commit_id=str(catalog.get("commit_id", "") or "").strip(),
        )

    def __post_init__(self) -> None:
        repository = self.repository.strip()
        branch = self.branch.strip()
        path = self.path.strip()
        commit_id = self.commit_id.strip()
        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "commit_id", commit_id)

        if not repository:
            raise ValueError("catalog.repository is required")
        if repository.startswith("-"):
            raise ValueError("catalog.repository must not start with '-'")
        if not branch:
            raise ValueError("catalog.branch is required")
        if not path:
            raise ValueError("catalog.path is required")
        catalog_path = PurePosixPath(path)
        if catalog_path.is_absolute():
            raise ValueError("catalog.path must be repository-relative")
        if ".." in catalog_path.parts:
            raise ValueError("catalog.path must not contain '..'")
        if commit_id and _COMMIT_ID_RE.fullmatch(commit_id) is None:
            raise ValueError("catalog.commit_id must be a hexadecimal Git commit id")

    def sync(self, cache_root: str | Path) -> CatalogSyncResult:
        """Fetch the branch, check out the selected revision, and read ``path``."""
        checkout = self.checkout_dir(cache_root)
        checkout.parent.mkdir(parents=True, exist_ok=True)
        if not checkout.exists():
            self._git(
                "clone",
                "--no-checkout",
                "--origin",
                "origin",
                self.repository,
                str(checkout),
            )
        elif not (checkout / ".git").is_dir():
            raise ConfigLoadError(
                self.repository,
                f"catalog repository cache is not a Git checkout: {checkout}",
            )

        remote_ref = f"refs/remotes/origin/{self.branch}"
        refspec = f"+refs/heads/{self.branch}:{remote_ref}"
        self._git("fetch", "--prune", "origin", refspec, cwd=checkout)
        branch_commit = self._git(
            "rev-parse",
            "--verify",
            f"{remote_ref}^{{commit}}",
            cwd=checkout,
        )
        resolved_commit = branch_commit
        if self.commit_id:
            resolved_commit = self._git(
                "rev-parse",
                "--verify",
                f"{self.commit_id}^{{commit}}",
                cwd=checkout,
            )
            ancestry = self._git_process(
                "merge-base",
                "--is-ancestor",
                resolved_commit,
                branch_commit,
                cwd=checkout,
            )
            if ancestry.returncode != 0:
                raise ConfigLoadError(
                    self.repository,
                    f"catalog.commit_id {self.commit_id!r} is not on branch {self.branch!r}",
                )

        self._git("checkout", "--force", "--detach", resolved_commit, cwd=checkout)
        self._git("clean", "-fdx", cwd=checkout)
        self._git("ls-files", "--error-unmatch", "--", self.path, cwd=checkout)

        selected_path = self.catalog_path(cache_root)
        checkout_root = checkout.resolve()
        try:
            selected_path.resolve().relative_to(checkout_root)
        except ValueError as exc:
            raise ConfigLoadError(
                self.repository,
                f"catalog.path escapes the repository checkout: {self.path}",
            ) from exc
        if not selected_path.is_file():
            raise ConfigLoadError(
                self.repository,
                f"catalog.path is not a file at {resolved_commit}: {self.path}",
            )

        return CatalogSyncResult(
            repository=self.repository,
            branch=self.branch,
            path=self.path,
            commit_id=self.commit_id,
            resolved_commit_id=resolved_commit,
            source_uri=selected_path.resolve().as_uri(),
            content=selected_path.read_text(encoding="utf-8"),
        )

    def checkout_dir(self, cache_root: str | Path) -> Path:
        """Return this source's stable local Git checkout directory."""
        repository_key = sha256(self.repository.encode()).hexdigest()
        return Path(cache_root) / "catalog" / "repositories" / repository_key

    def catalog_path(self, cache_root: str | Path) -> Path:
        """Return the selected catalog path inside the local checkout."""
        return self.checkout_dir(cache_root).joinpath(*PurePosixPath(self.path).parts)

    def _git(self, *args: str, cwd: Path | None = None) -> str:
        result = self._git_process(*args, cwd=cwd)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
            raise ConfigLoadError(
                self.repository,
                f"git {' '.join(args[:2])} failed: {detail}",
            )
        return result.stdout.strip()

    def _git_process(
        self,
        *args: str,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ConfigLoadError(self.repository, "git executable was not found") from exc
