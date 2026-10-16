"""SWE-bench payload validation and verifier shell construction."""

from __future__ import annotations

import json
import logging
import re
import shlex
from typing import Any

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from anexus.tasks.graders.base import GraderConfig

logger = logging.getLogger("anexus.tasks.graders.swebench")


class SWEInstallConfig(BaseModel):
    """Common install/test metadata carried by SWE-style task payloads."""

    model_config = ConfigDict(extra="allow")

    test_cmd: str | list[str] | None = None
    log_parser: str | None = None


class SWEMaterializedInstance(BaseModel):
    """Verifier-ready SWE instance shape expected by Anexus.

    This preserves common SWE-bench field names and accepts the two image key
    spellings seen in community data (`faas-image`) and Python code
    (`faas_image`).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    task_id: str
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str = ""
    patch: str = ""
    test_patch: str = ""
    fail_to_pass: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("FAIL_TO_PASS", "fail_to_pass"),
    )
    pass_to_pass: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("PASS_TO_PASS", "pass_to_pass"),
    )
    faas_image: str = Field(validation_alias=AliasChoices("faas-image", "faas_image"))
    workdir: str
    eval_script: str
    install_config: SWEInstallConfig | None = None
    version: str | None = None
    created_at: str | None = None
    hints_text: str = ""
    environment_setup_commit: str = ""
    language: str | None = None

    @field_validator(
        "task_id",
        "instance_id",
        "repo",
        "base_commit",
        "faas_image",
        "workdir",
        mode="before",
    )
    @classmethod
    def _non_empty_string(cls, value: object) -> str:
        text = "" if value is None else str(value).strip()
        if not text:
            raise ValueError("must be a non-empty string")
        return text

    @field_validator("eval_script", mode="before")
    @classmethod
    def _non_empty_script(cls, value: object) -> str:
        text = "" if value is None else str(value)
        if not text.strip():
            raise ValueError("must be a non-empty string")
        return text

    @field_validator("patch", "problem_statement", "test_patch", "hints_text", mode="before")
    @classmethod
    def _string_value(cls, value: object) -> str:
        return "" if value is None else str(value)

    @field_validator("fail_to_pass", "pass_to_pass", mode="before")
    @classmethod
    def _list_value(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return parse_json_list_or_singleton(value)
        if isinstance(value, list):
            return [str(item) for item in value if str(item)]
        return []


def payload_workdir(
    payload: dict[str, Any],
    data: dict[str, Any],
    config: GraderConfig,
) -> str:
    configured = _resolve_dot_path(
        payload,
        str(config.extra.get("workdir_from_payload", "metadata.workdir")),
    )
    return configured or str(data.get("workdir") or "").strip() or "/testbed"


def payload_base_commit(
    payload: dict[str, Any],
    data: dict[str, Any],
    config: GraderConfig,
) -> str:
    configured = _resolve_dot_path(
        payload,
        str(config.extra.get("base_commit_from_payload", "metadata.base_commit")),
    )
    return configured or str(data.get("base_commit") or "").strip()


def _resolve_dot_path(data: dict[str, Any], path: str) -> str | None:
    value: object = data
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, str) else None


def _merged_payload_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    return {**metadata, **payload} if isinstance(metadata, dict) else dict(payload)


def normalized_payload_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = _merged_payload_metadata(payload)
    if "faas_image" not in data and "faas-image" in data:
        data["faas_image"] = data["faas-image"]
    return data


def validated_materialized_instance(
    payload: dict[str, Any],
    *,
    emit_warning: bool = False,
) -> tuple[SWEMaterializedInstance | None, list[str]]:
    data = normalized_payload_data(payload)
    try:
        instance = SWEMaterializedInstance.model_validate(data)
    except ValidationError as exc:
        warnings = _validation_error_messages(exc)
        if not _should_emit_materialized_warning(data):
            return None, []
        if emit_warning:
            _log_materialized_warnings(data, warnings)
        return None, warnings

    warnings = _materialized_semantic_warnings(instance)
    if emit_warning and warnings:
        _log_materialized_warnings(data, warnings)
    return instance, warnings


def _validation_error_messages(exc: ValidationError) -> list[str]:
    messages: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ())) or "payload"
        messages.append(f"{loc}: {error.get('msg', 'invalid value')}")
    return messages


def _should_emit_materialized_warning(data: dict[str, Any]) -> bool:
    if isinstance(data.get("metadata"), dict):
        return True
    return any(str(data.get(key) or "").strip() for key in ("eval_script", "faas_image", "faas-image", "workdir"))


def _log_materialized_warnings(data: dict[str, Any], warnings: list[str]) -> None:
    if not warnings:
        return
    logger.warning(
        "SWE materialized payload validation warning task_id=%s instance_id=%s warnings=%s",
        data.get("task_id", ""),
        data.get("instance_id", ""),
        warnings,
    )


def _materialized_semantic_warnings(instance: SWEMaterializedInstance) -> list[str]:
    warnings: list[str] = []
    if not instance.workdir.startswith("/"):
        warnings.append("workdir should be an absolute sandbox path")
    if instance.patch.strip() and instance.patch.strip() in instance.eval_script:
        warnings.append("eval_script must not apply the gold patch from payload.patch")
    if instance.test_patch.strip() and instance.test_patch.strip() not in instance.eval_script:
        warnings.append("eval_script should apply payload.test_patch")
    if re.search(r"\bgit\s+reset\b", instance.eval_script):
        warnings.append("eval_script should not git reset after the model patch is applied")
    if (instance.fail_to_pass or instance.pass_to_pass) and not _install_config_log_parser(instance.install_config):
        warnings.append("FAIL_TO_PASS/PASS_TO_PASS are present but install_config.log_parser is missing")
    return warnings


def _install_config_log_parser(install_config: SWEInstallConfig | None) -> str:
    if install_config is None:
        return ""
    return str(install_config.log_parser or "").strip()


def validation_warning_details(warnings: list[str]) -> dict[str, Any]:
    return {"materialized_validation_warnings": warnings} if warnings else {}


def has_expected_tests(data: dict[str, Any]) -> bool:
    return bool(parse_list_field(data, "FAIL_TO_PASS") or parse_list_field(data, "PASS_TO_PASS"))


def exit_code_resolved(eval_output: str, eval_exit_code: int) -> bool:
    match = re.search(r"\bOMNIGRIL_EXIT_CODE=(\d+)\b", eval_output or "")
    if match:
        return int(match.group(1)) == 0
    return eval_exit_code == 0


def has_payload_eval_contract(data: dict[str, Any]) -> bool:
    """Return true when the payload itself defines how evaluation should run."""

    install_config = data.get("install_config")
    if isinstance(install_config, dict) and any(
        key in install_config
        for key in (
            "env_vars",
            "install",
            "log_parser",
            "packages",
            "pip_packages",
            "pre_install",
            "reqs_path",
            "test_cmd",
        )
    ):
        return True
    if isinstance(data.get("environment"), str) and data["environment"].strip():
        return True
    return bool(str(data.get("faas_image") or data.get("faas-image") or "").strip())


def parse_list_field(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if isinstance(value, str):
        value = parse_json_list_or_singleton(value)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def parse_json_list_or_singleton(text: str) -> list[str]:
    """Parse a JSON array string, or wrap as single-element list if not valid JSON array."""
    if not text.startswith("["):
        return [text] if text else []
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(decoded, list):
        return [str(item) for item in decoded]
    return [text]


def generic_test_command(data: dict[str, Any]) -> str:
    install_config = data.get("install_config")
    if isinstance(install_config, dict):
        test_cmd = str(install_config.get("test_cmd", "") or "").strip()
        if test_cmd:
            return test_cmd
    test_cmd = str(data.get("test_cmd", "") or "").strip()
    if test_cmd:
        return test_cmd
    return "pytest --no-header -rA --tb=line --color=no -p no:cacheprovider"


def payload_environment_lines(data: dict[str, Any]) -> list[str]:
    lines = _payload_env_var_lines(data)
    if _uses_prebuilt_verifier_image(data):
        lines.extend(_prebuilt_verifier_environment_lines())
        return lines

    prefix = _payload_environment_prefix(data)
    if prefix:
        env_name = _payload_environment_name(data)
        raw_environment = str(data.get("environment", "") or "").rstrip("\n")
        lines.extend(
            [
                f"ANEXUS_ENV_PREFIX={shlex.quote(prefix)}",
                f"ANEXUS_ENV_NAME={shlex.quote(env_name)}",
                "ANEXUS_ENV_CREATED=0",
                "ANEXUS_CONDA=",
                'export CONDA_REMOTE_CONNECT_TIMEOUT_SECS="${CONDA_REMOTE_CONNECT_TIMEOUT_SECS:-30}"',
                'export CONDA_REMOTE_READ_TIMEOUT_SECS="${CONDA_REMOTE_READ_TIMEOUT_SECS:-90}"',
                'if [ ! -x "$ANEXUS_ENV_PREFIX/bin/python" ]; then',
                '  ANEXUS_CONDA="$(command -v conda || true)"',
                '  if [ -z "$ANEXUS_CONDA" ]; then',
                "    for candidate in /opt/conda/bin/conda "
                "/opt/miniconda3/bin/conda /opt/miniconda/bin/conda "
                "/usr/local/bin/conda; do",
                '      if [ -x "$candidate" ]; then',
                '        ANEXUS_CONDA="$candidate"',
                "        break",
                "      fi",
                "    done",
                "  fi",
                '  if [ -z "$ANEXUS_CONDA" ]; then',
                '    echo "payload environment requires conda, but conda is unavailable" >&2',
                "    exit 127",
                "  fi",
                '  if [ -n "$ANEXUS_ENV_NAME" ]; then',
                (
                    '    ANEXUS_EXISTING_ENV="$("$ANEXUS_CONDA" env list '
                    '| awk -v name="$ANEXUS_ENV_NAME" \'$1 == name {print $NF; exit}\')"'
                ),
                '    if [ -n "$ANEXUS_EXISTING_ENV" ] && [ -x "$ANEXUS_EXISTING_ENV/bin/python" ]; then',
                '      ANEXUS_ENV_PREFIX="$ANEXUS_EXISTING_ENV"',
                "    fi",
                "  fi",
                "fi",
                'if [ ! -x "$ANEXUS_ENV_PREFIX/bin/python" ]; then',
                '  mkdir -p "$(dirname "$ANEXUS_ENV_PREFIX")"',
                "  cat > /tmp/anexus_environment.yml <<'EOF_ANEXUS_ENVIRONMENT'",
                raw_environment,
                "EOF_ANEXUS_ENVIRONMENT",
                "  ANEXUS_CONDA_CREATE_ATTEMPT=1",
                '  until "$ANEXUS_CONDA" env create -f /tmp/anexus_environment.yml; do',
                '    if [ "$ANEXUS_CONDA_CREATE_ATTEMPT" -ge 3 ]; then',
                '      echo "payload conda environment create failed after $ANEXUS_CONDA_CREATE_ATTEMPT attempts" >&2',
                "      exit 127",
                "    fi",
                '    echo "payload conda environment create failed; retrying $ANEXUS_CONDA_CREATE_ATTEMPT/3" >&2',
                "    sleep $((ANEXUS_CONDA_CREATE_ATTEMPT * 15))",
                "    ANEXUS_CONDA_CREATE_ATTEMPT=$((ANEXUS_CONDA_CREATE_ATTEMPT + 1))",
                "  done",
                "  ANEXUS_ENV_CREATED=1",
                "fi",
                'if [ ! -x "$ANEXUS_ENV_PREFIX/bin/python" ] && [ -n "$ANEXUS_ENV_NAME" ]; then',
                (
                    '  ANEXUS_CREATED_ENV="$("$ANEXUS_CONDA" env list '
                    '| awk -v name="$ANEXUS_ENV_NAME" \'$1 == name {print $NF; exit}\')"'
                ),
                '  if [ -n "$ANEXUS_CREATED_ENV" ] && [ -x "$ANEXUS_CREATED_ENV/bin/python" ]; then',
                '    ANEXUS_ENV_PREFIX="$ANEXUS_CREATED_ENV"',
                "  fi",
                "fi",
                'if [ ! -x "$ANEXUS_ENV_PREFIX/bin/python" ]; then',
                '  echo "payload environment was not created at a runnable prefix: $ANEXUS_ENV_PREFIX" >&2',
                "  exit 127",
                "fi",
                'export PATH="$ANEXUS_ENV_PREFIX/bin:$PATH"',
                'export CONDA_PREFIX="$ANEXUS_ENV_PREFIX"',
                "export PYTHONNOUSERSITE=1",
                "hash -r || true",
            ]
        )
    setup_commands = _payload_setup_commands(data)
    if setup_commands:
        lines.extend(_checked_shell_command_lines(setup_commands))
    return lines


def _uses_prebuilt_verifier_image(data: dict[str, Any]) -> bool:
    return bool(str(data.get("faas_image") or data.get("faas-image") or "").strip())


def _prebuilt_verifier_environment_lines() -> list[str]:
    return [
        "if [ -x /opt/miniconda3/envs/testbed/bin/python ]; then",
        '  export PATH="/opt/miniconda3/envs/testbed/bin:$PATH"',
        '  export CONDA_PREFIX="/opt/miniconda3/envs/testbed"',
        '  export CONDA_DEFAULT_ENV="testbed"',
        "fi",
        "export PYTHONNOUSERSITE=1",
        "hash -r || true",
    ]


def _payload_env_var_lines(data: dict[str, Any]) -> list[str]:
    install_config = data.get("install_config")
    if not isinstance(install_config, dict):
        return []
    env_vars = install_config.get("env_vars")
    if not isinstance(env_vars, dict):
        return []

    lines: list[str] = []
    for key, value in sorted(env_vars.items()):
        name = str(key).strip()
        if not name.isidentifier():
            continue
        lines.append(f"export {name}={shlex.quote(str(value))}")
    return lines


def _checked_shell_command_lines(commands: list[str]) -> list[str]:
    lines: list[str] = []
    for command in commands:
        lines.extend(
            [
                "if ! (",
                *[f"  {line}" for line in command.splitlines()],
                "); then",
                f"  echo {shlex.quote(f'payload setup command failed: {command}')} >&2",
                "  exit 127",
                "fi",
            ]
        )
    return lines


def _payload_setup_commands(data: dict[str, Any]) -> list[str]:
    install_config = data.get("install_config")
    if not isinstance(install_config, dict):
        return []

    commands: list[str] = []
    commands.extend(_shell_commands_from_payload(install_config.get("pre_install")))
    if not _payload_environment_prefix(data):
        commands.extend(_requirements_install_commands(install_config))

        pip_packages = _shell_values_from_payload(install_config.get("pip_packages"))
        if pip_packages:
            packages = " ".join(shlex.quote(package) for package in pip_packages)
            commands.append(f"pip install {packages}")

    commands.extend(_shell_commands_from_payload(install_config.get("install")))
    return commands


def _requirements_install_commands(install_config: dict[str, Any]) -> list[str]:
    paths = _shell_values_from_payload(install_config.get("reqs_path"))
    packages = str(install_config.get("packages", "") or "").strip()
    if _looks_like_requirements_path(packages):
        paths.append(packages)

    commands: list[str] = []
    for path in dedupe_preserve_order(paths):
        quoted_path = shlex.quote(path)
        commands.append(
            "\n".join(
                [
                    f"if [ ! -f {quoted_path} ]; then",
                    f"  echo {shlex.quote(f'payload requirements file missing: {path}')} >&2",
                    "  exit 127",
                    "fi",
                    f"pip install -r {quoted_path}",
                ]
            )
        )
    return commands


def _looks_like_requirements_path(value: str) -> bool:
    if not value or any(char.isspace() for char in value):
        return False
    lowered = value.lower()
    return "requirements" in lowered or lowered.endswith(".txt") or lowered.endswith(".pip") or lowered.endswith(".req")


def _payload_environment_prefix(data: dict[str, Any]) -> str:
    parsed = _payload_environment_mapping(data)
    if not parsed:
        return ""
    install_config = data.get("install_config")
    if isinstance(install_config, dict) and install_config.get("no_use_env"):
        return ""

    prefix = parsed.get("prefix")
    if not isinstance(prefix, str):
        return ""
    return prefix.strip()


def _payload_environment_name(data: dict[str, Any]) -> str:
    parsed = _payload_environment_mapping(data)
    if not parsed:
        return ""
    name = parsed.get("name")
    if not isinstance(name, str):
        return ""
    return name.strip()


def _payload_environment_mapping(data: dict[str, Any]) -> dict[str, Any]:
    raw_environment = data.get("environment")
    if not isinstance(raw_environment, str) or not raw_environment.strip():
        return {}
    try:
        parsed = yaml.safe_load(raw_environment)
    except yaml.YAMLError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _shell_commands_from_payload(value: object) -> list[str]:
    if isinstance(value, str):
        command = value.strip()
        return [command] if command else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _shell_values_from_payload(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def test_patch_checkout_paths(test_patch: str) -> list[str]:
    paths: list[str] = []
    current_old_path = ""
    for line in test_patch.splitlines():
        if line.startswith("diff --git "):
            current_old_path = ""
            try:
                parts = shlex.split(line)
            except ValueError:
                continue
            if len(parts) < 4:
                continue
            current_old_path = _strip_git_patch_prefix(parts[2])
            if current_old_path == "/dev/null":
                current_old_path = ""
            continue
        if not line.startswith("--- "):
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        if len(parts) < 2:
            continue
        path = _strip_git_patch_prefix(parts[1])
        if path == "/dev/null":
            current_old_path = ""
            continue
        if path and path != "/dev/null":
            current_old_path = path
        if current_old_path:
            paths.append(current_old_path)
    return dedupe_preserve_order(paths)


def _strip_git_patch_prefix(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
