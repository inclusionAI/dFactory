"""KGB trace exporter runtime patching.

The KGB image already owns trace export execution via
``/opt/trace_export/trace_exporter_cron.py``.  Until the image can be rebuilt,
Anexus installs a patched copy of ``trace_export.py`` into each KGB sandbox
after startup and then lets the packaged cron wrapper call it normally.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources

TRACE_EXPORT_TARGET = "/opt/trace_export/trace_export.py"


def trace_export_script() -> str:
    """Return the patched KGB trace exporter script packaged with Anexus."""

    return resources.files("anexus.gateway.kgb.assets").joinpath("trace_export.py").read_text(encoding="utf-8")


def install_trace_export_command() -> str:
    """Return a sandbox command that replaces KGB's trace exporter in place."""

    script = trace_export_script()
    digest = hashlib.sha256(script.encode("utf-8")).hexdigest()
    return "\n".join(
        [
            "python - <<'PY'",
            "from pathlib import Path",
            f"script = {json.dumps(script)}",
            f"digest = {json.dumps(digest)}",
            f"target = Path({json.dumps(TRACE_EXPORT_TARGET)})",
            "target.parent.mkdir(parents=True, exist_ok=True)",
            "target.write_text(script, encoding='utf-8')",
            "target.chmod(0o755)",
            "print(f'[anexus] installed patched KGB trace_export.py sha256={digest}')",
            "PY",
        ]
    )
