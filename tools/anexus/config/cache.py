"""Load catalogs directly from their local Git checkout."""

from __future__ import annotations

import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

from anexus.config.catalog import Catalog, parse_catalog
from anexus.config.catalog_source import CatalogSource, CatalogSyncResult
from anexus.config.loader import ConfigLoadError, parse_config_text

_LEGACY_SNAPSHOT_NAMES = ("selected.catalog.data", "selected.catalog.meta.json")


@dataclass(frozen=True)
class CachedCatalog:
    catalog: Catalog
    content: str
    source_uri: str
    locally_modified: bool = False


class CatalogCache:
    """Read the selected catalog from a Git checkout managed by ``update``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @property
    def cache_dir(self) -> Path:
        return self.root / "catalog"

    def update(self, result: CatalogSyncResult) -> CachedCatalog:
        """Validate a synchronized catalog and discard obsolete snapshots."""
        data = parse_config_text(result.content, source_uri=result.source_uri)
        catalog = parse_catalog(data)
        self._discard_legacy_snapshots()
        return CachedCatalog(
            catalog=catalog,
            content=result.content,
            source_uri=result.source_uri,
        )

    def load(self, source: CatalogSource) -> CachedCatalog | None:
        """Load local catalog contents, warning without rejecting local edits."""
        checkout = source.checkout_dir(self.root)
        catalog_path = source.catalog_path(self.root)
        if not catalog_path.is_file():
            return None

        try:
            catalog_path.resolve().relative_to(checkout.resolve())
        except ValueError as exc:
            raise ConfigLoadError(
                catalog_path.absolute().as_uri(),
                "local catalog path escapes its Git checkout; run anexus update",
            ) from exc

        source_uri = catalog_path.resolve().as_uri()
        try:
            content = catalog_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigLoadError(
                source_uri,
                f"could not read local catalog: {exc}; run anexus update",
            ) from exc

        modifications = _repository_modifications(checkout)
        if modifications:
            summary = ", ".join(modifications[:3])
            if len(modifications) > 3:
                summary = f"{summary}, ..."
            warnings.warn(
                "local catalog checkout contains modifications; using them as-is. "
                f"Inspect {checkout} before running 'anexus update', which replaces local "
                f"changes. Changes: {summary}",
                UserWarning,
                stacklevel=2,
            )

        data = parse_config_text(content, source_uri=source_uri)
        return CachedCatalog(
            catalog=parse_catalog(data),
            content=content,
            source_uri=source_uri,
            locally_modified=bool(modifications),
        )

    def _discard_legacy_snapshots(self) -> None:
        for name in _LEGACY_SNAPSHOT_NAMES:
            path = self.cache_dir / name
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                warnings.warn(
                    f"could not remove obsolete catalog snapshot {path}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )


def _repository_modifications(checkout: Path) -> tuple[str, ...]:
    if not (checkout / ".git").exists():
        return ()
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=checkout,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        warnings.warn(
            f"could not inspect local catalog checkout {checkout}: {exc}; using it as-is",
            UserWarning,
            stacklevel=2,
        )
        return ()
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        warnings.warn(
            f"could not inspect local catalog checkout {checkout}: {detail}; using it as-is",
            UserWarning,
            stacklevel=2,
        )
        return ()
    return tuple(line for line in result.stdout.splitlines() if line.strip())
