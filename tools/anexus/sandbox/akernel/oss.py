"""In-sandbox OSS transfer command construction."""

from __future__ import annotations

import shlex
from urllib.parse import unquote, urlsplit

from anexus.config.catalog import ObjectStoreSpec
from anexus.sandbox.provider import (
    ArtifactMapping,
    ExecResult,
    FixtureMapping,
    SandboxLease,
)


def oss_download_command(
    fixture: FixtureMapping,
    *,
    cli_path: str,
    object_store: ObjectStoreSpec,
) -> str:
    """Build a command that downloads one fixture inside the sandbox."""

    source_uri = fixture.source_uri.strip()
    target_path = fixture.target_path.strip()
    if not source_uri:
        raise ValueError("fixture source_uri is required")
    if not target_path:
        raise ValueError("fixture target_path is required")
    if fixture.kind == "directory":
        script = oss_directory_download_script(source_uri, target_path, cli_path=cli_path, object_store=object_store)
    else:
        script = oss_file_download_script(source_uri, target_path, cli_path=cli_path, object_store=object_store)
    return "bash -lc " + shlex.quote(script)


def oss_upload_command(
    artifact: ArtifactMapping,
    *,
    cli_path: str,
    object_store: ObjectStoreSpec,
) -> str:
    """Build a command that uploads one artifact inside the sandbox."""

    source_path = artifact.source_path.strip()
    target_uri = artifact.target_uri.strip()
    if not source_path:
        raise ValueError("artifact source_path is required")
    if not target_uri:
        raise ValueError("artifact target_uri is required")
    if artifact.kind == "directory":
        script = oss_directory_upload_script(source_path, target_uri, cli_path=cli_path, object_store=object_store)
    else:
        script = oss_file_upload_script(source_path, target_uri, cli_path=cli_path, object_store=object_store)
    return "bash -lc " + shlex.quote(script)


def oss_directory_download_script(
    source_uri: str,
    target_path: str,
    *,
    cli_path: str,
    object_store: ObjectStoreSpec,
) -> str:
    """Build the shell script for downloading an OSS directory."""

    bucket, prefix = parse_oss_uri(source_uri)
    flags = _oss_flags(object_store)
    return "\n".join(
        [
            *_oss_script_prelude(cli_path),
            f"DT_OSS_BUCKET={shlex.quote(bucket)}",
            f"DT_OSS_PREFIX={shlex.quote(prefix)}",
            f"DT_OSS_LOCAL={shlex.quote(target_path)}",
            "export DT_OSS_BUCKET DT_OSS_PREFIX DT_OSS_LOCAL",
            'LOCAL_DIR="${DT_OSS_LOCAL:-/opt/workspace}"',
            'REMOTE_DIR="oss://${DT_OSS_BUCKET}"',
            'if [ -n "${DT_OSS_PREFIX}" ]; then',
            '  REMOTE_DIR="${REMOTE_DIR}/${DT_OSS_PREFIX%/}"',
            "fi",
            'REMOTE_DIR="${REMOTE_DIR%/}/"',
            'mkdir -p "$LOCAL_DIR"',
            f'"$OSSUTIL" cp "$REMOTE_DIR" "$LOCAL_DIR/" -r -f {flags} 2>&1',
            'echo "[oss-dl] downloaded directory from ${REMOTE_DIR} -> ${LOCAL_DIR}/"',
        ]
    )


def oss_file_download_script(
    source_uri: str,
    target_path: str,
    *,
    cli_path: str,
    object_store: ObjectStoreSpec,
) -> str:
    """Build the shell script for downloading one OSS object."""

    parse_oss_uri(source_uri)
    flags = _oss_flags(object_store)
    return "\n".join(
        [
            *_oss_script_prelude(cli_path),
            f"SOURCE_URI={shlex.quote(source_uri)}",
            f"TARGET_PATH={shlex.quote(target_path)}",
            'mkdir -p "$(dirname "$TARGET_PATH")"',
            f'"$OSSUTIL" cp "$SOURCE_URI" "$TARGET_PATH" -f {flags} 2>&1',
            'SIZE=$(wc -c < "$TARGET_PATH" 2>/dev/null | tr -d " ")',
            'echo "[oss-dl] downloaded ${SIZE:-0} bytes from ${SOURCE_URI} -> ${TARGET_PATH}"',
        ]
    )


def oss_directory_upload_script(
    source_path: str,
    target_uri: str,
    *,
    cli_path: str,
    object_store: ObjectStoreSpec,
) -> str:
    """Build the shell script for uploading a directory to OSS."""

    parse_oss_uri(target_uri)
    flags = _oss_flags(object_store)
    return "\n".join(
        [
            *_oss_script_prelude(cli_path),
            f"SOURCE_PATH={shlex.quote(source_path)}",
            f"TARGET_URI={shlex.quote(target_uri)}",
            'TARGET_DIR="${TARGET_URI%/}/"',
            f'"$OSSUTIL" cp "$SOURCE_PATH" "$TARGET_DIR" -r -f {flags} 2>&1',
            'echo "[oss-up] uploaded directory from ${SOURCE_PATH} -> ${TARGET_DIR}"',
        ]
    )


def oss_file_upload_script(
    source_path: str,
    target_uri: str,
    *,
    cli_path: str,
    object_store: ObjectStoreSpec,
) -> str:
    """Build the shell script for uploading one file to OSS."""

    parse_oss_uri(target_uri)
    flags = _oss_flags(object_store)
    return "\n".join(
        [
            *_oss_script_prelude(cli_path),
            f"SOURCE_PATH={shlex.quote(source_path)}",
            f"TARGET_URI={shlex.quote(target_uri)}",
            f'"$OSSUTIL" cp "$SOURCE_PATH" "$TARGET_URI" -f {flags} 2>&1',
            'SIZE=$(wc -c < "$SOURCE_PATH" 2>/dev/null | tr -d " ")',
            'echo "[oss-up] uploaded ${SIZE:-0} bytes from ${SOURCE_PATH} -> ${TARGET_URI}"',
        ]
    )


def copy_between_temp_uri(
    source_lease: SandboxLease,
    path_hash: str,
    *,
    object_store: ObjectStoreSpec,
) -> str:
    """Build the real OSS URI used to relay data between sandboxes."""

    bucket = object_store.bucket
    prefix = object_store.prefix
    object_path = "/".join(
        part.strip("/") for part in (prefix, "cross-sandbox", source_lease.sandbox_id, path_hash) if part.strip("/")
    )
    return f"oss://{bucket}/{object_path}"


def exec_failure_summary(result: ExecResult) -> str:
    """Return the most useful command failure text."""

    return result.stderr.strip() or result.stdout.strip() or f"exit_code={result.exit_code}"


def parse_oss_uri(uri: str) -> tuple[str, str]:
    parts = urlsplit(uri)
    if parts.scheme != "oss" or not parts.netloc:
        raise ValueError(f"OSS URI must use oss://bucket/path: {uri}")
    return parts.netloc, unquote(parts.path.lstrip("/"))


def _oss_script_prelude(cli_path: str) -> list[str]:
    return [
        "set -eo pipefail",
        f"OSSUTIL={shlex.quote(cli_path)}",
        'if [ ! -x "$OSSUTIL" ]; then',
        '  echo "[oss] missing ossutil: $OSSUTIL" >&2',
        "  exit 127",
        "fi",
    ]


def _oss_flags(object_store: ObjectStoreSpec) -> str:
    flags = (
        f"--endpoint={shlex.quote(object_store.endpoint)} "
        f"--access-key-id={shlex.quote(object_store.access_key_id)} "
        f"--access-key-secret={shlex.quote(object_store.access_key_secret)} "
        f"--sign-version {shlex.quote(object_store.sign_version)}"
    )
    if object_store.no_progress:
        flags += " --no-progress"
    return flags
