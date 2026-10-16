"""SWE-bench grader implementation."""

from __future__ import annotations

import base64
import json
import logging
import shlex
from typing import Any

from anexus.tasks.graders import swebench_payload as payloads
from anexus.tasks.graders import swebench_results as results
from anexus.tasks.graders.base import ArtifactSandbox, BaseGrader, GraderConfig, GraderResult

logger = logging.getLogger(__name__)

START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

SWE_BENCH_APPLY_COMMANDS = (
    "git apply --verbose /tmp/agent_patch.diff",
    "git apply --verbose --reject /tmp/agent_patch.diff",
    "patch --batch --fuzz=5 -p1 -i /tmp/agent_patch.diff",
)

SWEInstallConfig = payloads.SWEInstallConfig
SWEMaterializedInstance = payloads.SWEMaterializedInstance


class SWEBenchGrader(BaseGrader):
    """SWE-bench patch extraction, eval-script generation, and grading."""

    name = "swebench"

    async def extract_artifact(
        self,
        sandbox: ArtifactSandbox,
        payload: dict[str, Any],
        config: GraderConfig | None = None,
    ) -> str:
        """Extract an agent patch from the task sandbox.

        The extraction order follows lumen-runner and SWE-bench practice:
        first read `/tmp/solution.patch`, then fall back to `git diff
        <base_commit>` from the task workdir.
        """

        grader_config = config or GraderConfig()
        data = payloads.normalized_payload_data(payload)
        workdir = payloads.payload_workdir(payload, data, grader_config)
        base_commit = payloads.payload_base_commit(payload, data, grader_config)

        try:
            patch = await sandbox.read_file("/tmp/solution.patch")
            if patch and patch.strip():
                return patch
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "failed to read SWE-bench solution patch "
                "path=/tmp/solution.patch workdir=%s base_commit=%s "
                "sandbox_id=%s instance_id=%s error_type=%s error=%s",
                workdir,
                base_commit,
                getattr(sandbox, "sandbox_id", ""),
                getattr(sandbox, "instance_id", ""),
                type(exc).__name__,
                exc,
            )

        if not base_commit:
            return ""

        try:
            result = await sandbox.run_command(
                f"cd {shlex.quote(workdir)} && git diff --no-pager {shlex.quote(base_commit)}",
                timeout=30,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "failed to extract SWE-bench artifact from git diff "
                "workdir=%s base_commit=%s sandbox_id=%s instance_id=%s "
                "error_type=%s error=%s",
                workdir,
                base_commit,
                getattr(sandbox, "sandbox_id", ""),
                getattr(sandbox, "instance_id", ""),
                type(exc).__name__,
                exc,
            )
            return ""

        if result.exit_code == 0 and result.stdout and result.stdout.strip():
            return result.stdout
        if result.exit_code != 0:
            logger.warning(
                "failed to extract SWE-bench artifact from git diff "
                "workdir=%s base_commit=%s sandbox_id=%s instance_id=%s "
                "exit_code=%s stderr=%s",
                workdir,
                base_commit,
                getattr(sandbox, "sandbox_id", ""),
                getattr(sandbox, "instance_id", ""),
                result.exit_code,
                (result.stderr or "")[:500],
            )
        return ""

    def generate_eval_script(
        self,
        payload: dict[str, Any],
        config: GraderConfig | None = None,
    ) -> str:
        """Generate an eval script on the host side, with deterministic fallback."""

        data = payloads.normalized_payload_data(payload)
        materialized, _warnings = payloads.validated_materialized_instance(payload, emit_warning=True)
        if bool(data.get("run_script")) and bool(data.get("parsing_script")):
            return self._generate_pro_eval_script(data)

        if materialized is not None:
            return materialized.eval_script

        explicit_eval_script = str(data.get("eval_script", "") or "")
        if explicit_eval_script.strip():
            return explicit_eval_script

        if payloads.has_payload_eval_contract(data):
            generic_script = self._generate_generic_pytest_eval_script(data)
            if generic_script.strip():
                return generic_script

        repo = str(data.get("repo", ""))
        version = str(data.get("version", ""))
        base_commit = str(data.get("base_commit", ""))
        test_patch = str(data.get("test_patch", ""))
        if repo and version and base_commit and test_patch:
            standard_script = self._generate_standard_eval_script(
                data,
                repo=repo,
                version=version,
                base_commit=base_commit,
                test_patch=test_patch,
            )
            if standard_script.strip():
                return standard_script

        return self._generate_generic_pytest_eval_script(data)

    def build_command(
        self,
        payload: dict[str, Any],
        config: GraderConfig | None = None,
    ) -> str:
        """Build the verifier sandbox command: reset, apply patch, run eval."""

        grader_config = config or GraderConfig()
        data = payloads.normalized_payload_data(payload)
        payloads.validated_materialized_instance(payload, emit_warning=True)
        workdir = payloads.payload_workdir(payload, data, grader_config)
        base_commit = payloads.payload_base_commit(payload, data, grader_config)

        parts = [f"cd {shlex.quote(workdir)}"]
        if base_commit:
            parts.append(f"git reset --hard {shlex.quote(base_commit)} && git clean -fdx")
        apply_chain = " || ".join(f"({command})" for command in SWE_BENCH_APPLY_COMMANDS)
        parts.append(f"({apply_chain})")
        parts.append("bash /tmp/eval_script.sh 2>&1")
        return " && ".join(parts)

    def grade_output(
        self,
        eval_output: str,
        eval_exit_code: int,
        payload: dict[str, Any],
        config: GraderConfig | None = None,
    ) -> GraderResult:
        """Grade eval output and fall back to exit-code semantics if needed."""

        data = payloads.normalized_payload_data(payload)
        materialized, validation_warnings = payloads.validated_materialized_instance(
            payload,
            emit_warning=True,
        )
        if materialized is not None and not payloads.has_expected_tests(data):
            resolved = payloads.exit_code_resolved(eval_output, eval_exit_code)
            return GraderResult(
                score=1.0 if resolved else 0.0,
                passed=resolved,
                error="" if resolved else "materialized_eval_failed",
                eval_log=eval_output[-4096:] if eval_output else "",
                details={
                    "eval_exit_code": eval_exit_code,
                    "resolution_status": "RESOLVED_FULL" if resolved else "RESOLVED_NO",
                    "grading_mode": "materialized_exit_code",
                    **payloads.validation_warning_details(validation_warnings),
                },
            )

        if bool(data.get("run_script")) and bool(data.get("parsing_script")):
            grading = self._grade_pro_output(eval_output, data)
        else:
            grading = self._grade_standard_output(eval_output, data)

        if "error" in grading and not grading.get("resolved", False):
            resolved = eval_exit_code == 0
            fallback_reason = str(grading.get("error", ""))
            return GraderResult(
                score=1.0 if resolved else 0.0,
                passed=resolved,
                error="" if resolved else fallback_reason,
                eval_log=eval_output[-4096:] if eval_output else "",
                details={
                    "eval_exit_code": eval_exit_code,
                    "resolution_status": "RESOLVED_FULL" if resolved else "RESOLVED_NO",
                    "grading_fallback": True,
                    "grading_fallback_reason": fallback_reason,
                    **payloads.validation_warning_details(validation_warnings),
                },
            )

        resolved = bool(grading.get("resolved", False))
        return GraderResult(
            score=1.0 if resolved else 0.0,
            passed=resolved,
            eval_log=eval_output[-4096:] if eval_output else "",
            details={
                "eval_exit_code": eval_exit_code,
                "resolution_status": str(grading.get("resolution_status", "")),
                "f2p_rate": float(grading.get("f2p_rate", 0.0)),
                "p2p_rate": float(grading.get("p2p_rate", 0.0)),
                "f2p_success": list(grading.get("f2p_success", [])),
                "f2p_failure": list(grading.get("f2p_failure", [])),
                "p2p_success": list(grading.get("p2p_success", [])),
                "p2p_failure": list(grading.get("p2p_failure", [])),
                **payloads.validation_warning_details(validation_warnings),
            },
        )

    @staticmethod
    def _strip_install_commands(script_lines: list[str]) -> list[str]:
        filtered: list[str] = []
        for line in script_lines:
            stripped = line.strip()
            if "pip install" in stripped or stripped.startswith("conda install"):
                continue
            filtered.append(line)
        return filtered

    def _generate_standard_eval_script(
        self,
        data: dict[str, Any],
        *,
        repo: str,
        version: str,
        base_commit: str,
        test_patch: str,
    ) -> str:
        try:
            from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
            from swebench.harness.test_spec.create_scripts import make_eval_script_list
        except ImportError as exc:
            logger.warning(
                "failed to import SWE-bench eval script generator task_id=%s repo=%s version=%s error_type=%s error=%s",
                data.get("task_id", ""),
                repo,
                version,
                type(exc).__name__,
                exc,
            )
            return ""

        repo_specs = MAP_REPO_VERSION_TO_SPECS.get(repo)
        if not repo_specs or version not in repo_specs:
            logger.debug(
                "SWE-bench repo/version specs unavailable; using generic eval fallback task_id=%s repo=%s version=%s",
                data.get("task_id", ""),
                repo,
                version,
            )
            return ""

        try:
            env_name = "testbed"
            repo_directory = f"/{env_name}"
            lines = make_eval_script_list(
                data,
                repo_specs[version],
                env_name,
                repo_directory,
                base_commit,
                test_patch,
            )
            lines = self._strip_install_commands(lines)
            return "\n".join(["#!/bin/bash", "set -uxo pipefail", *lines]) + "\n"
        except (ValueError, TypeError, IndexError, AttributeError) as exc:
            logger.warning(
                "failed to generate standard SWE-bench eval script "
                "task_id=%s repo=%s version=%s base_commit=%s "
                "error_type=%s error=%s",
                data.get("task_id", ""),
                repo,
                version,
                base_commit,
                type(exc).__name__,
                exc,
            )
            return ""

    def _generate_generic_pytest_eval_script(self, data: dict[str, Any]) -> str:
        tests = payloads.dedupe_preserve_order(
            [
                *payloads.parse_list_field(data, "FAIL_TO_PASS"),
                *payloads.parse_list_field(data, "PASS_TO_PASS"),
            ]
        )
        if not tests:
            return ""
        test_cmd = payloads.generic_test_command(data)
        test_args = " ".join(shlex.quote(test) for test in tests)
        test_patch = str(data.get("test_patch", "") or "")
        base_commit = str(data.get("base_commit", "") or "")

        lines = ["#!/bin/bash", "set -uxo pipefail"]
        lines.extend(payloads.payload_environment_lines(data))
        test_files = payloads.test_patch_checkout_paths(test_patch)
        if base_commit and test_files:
            files = " ".join(shlex.quote(path) for path in test_files)
            lines.append(f"git checkout {shlex.quote(base_commit)} -- {files}")
        if test_patch.strip():
            lines.extend(
                [
                    "cat > /tmp/test.patch <<'EOF_ANEXUS_TEST_PATCH'",
                    test_patch.rstrip("\n"),
                    "EOF_ANEXUS_TEST_PATCH",
                    "git apply -v /tmp/test.patch",
                ]
            )
        lines.extend(
            [
                f'echo "{START_TEST_OUTPUT}"',
                f"{test_cmd} {test_args}".strip(),
                "status=$?",
                f'echo "{END_TEST_OUTPUT}"',
                "exit $status",
            ]
        )
        return "\n".join(lines) + "\n"

    def _generate_pro_eval_script(self, data: dict[str, Any]) -> str:
        run_script = str(data["run_script"])
        parsing_script = str(data["parsing_script"])
        selected = data.get("selected_test_files_to_run", [])
        if isinstance(selected, str):
            selected = payloads.parse_json_list_or_singleton(selected)
        selected_arg = ",".join(str(item) for item in selected)

        env_lines = self._extract_env_exports(str(data.get("base_dockerfile", "")))
        env_lines.extend(self._extract_env_exports(str(data.get("instance_dockerfile", ""))))
        env_block = "\n".join(env_lines)
        before_repo_set_cmd = str(data.get("before_repo_set_cmd", "") or "").strip()
        last_before_line = before_repo_set_cmd.split("\n")[-1] if before_repo_set_cmd else ""
        run_script_b64 = base64.b64encode(run_script.encode()).decode()
        parser_b64 = base64.b64encode(parsing_script.encode()).decode()
        return f"""#!/bin/bash
set -uxo pipefail
{env_block}

echo {run_script_b64} | base64 -d > /tmp/run_script.sh
echo {parser_b64} | base64 -d > /tmp/parser.py
chmod +x /tmp/run_script.sh

cd /app

{last_before_line}

echo "{START_TEST_OUTPUT}"
bash /tmp/run_script.sh {selected_arg} > /tmp/stdout.log 2> /tmp/stderr.log || true
python /tmp/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json || true
cat /tmp/output.json 2>/dev/null || echo '{{"tests": []}}'
echo ""
echo "{END_TEST_OUTPUT}"
"""

    def _grade_pro_output(self, eval_output: str, data: dict[str, Any]) -> dict[str, Any]:
        if START_TEST_OUTPUT in eval_output and END_TEST_OUTPUT in eval_output:
            payload_text = eval_output.split(START_TEST_OUTPUT, 1)[1].split(
                END_TEST_OUTPUT,
                1,
            )[0]
        else:
            payload_text = eval_output
        try:
            start = payload_text.index("{")
            end = payload_text.rindex("}") + 1
            parsed = json.loads(payload_text[start:end])
        except (ValueError, json.JSONDecodeError) as exc:
            return {"resolved": False, "error": f"parser_output_unreadable: {exc}"}

        status_map = {
            test.get("name"): str(test.get("status") or "").upper()
            for test in parsed.get("tests", []) or []
            if isinstance(test, dict) and test.get("name")
        }
        return results.grade_status_map(status_map, data)

    def _grade_standard_output(self, eval_output: str, data: dict[str, Any]) -> dict[str, Any]:
        try:
            from types import SimpleNamespace

            from swebench.harness.constants import (
                FAIL_ONLY_REPOS,
                FAIL_TO_PASS,
                MAP_REPO_VERSION_TO_SPECS,
                PASS_TO_PASS,
                EvalType,
            )
            from swebench.harness.grading import (
                ResolvedStatus,
                compute_fail_to_pass,
                compute_pass_to_pass,
                get_eval_tests_report,
                get_resolution_status,
            )
            from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

            repo = str(data.get("repo", ""))
            test_spec = SimpleNamespace(
                instance_id=str(data.get("instance_id", "")),
                repo=repo,
                version=str(data.get("version", "")),
                FAIL_TO_PASS=payloads.parse_list_field(data, "FAIL_TO_PASS"),
                PASS_TO_PASS=payloads.parse_list_field(data, "PASS_TO_PASS"),
            )
            repo_specs = MAP_REPO_VERSION_TO_SPECS.get(repo)
            has_standard_specs = bool(repo_specs and test_spec.version in repo_specs)
            if payloads.has_payload_eval_contract(data):
                parser = results.configured_log_parser(data) or MAP_REPO_TO_PARSER.get(repo)
            elif has_standard_specs:
                parser = MAP_REPO_TO_PARSER.get(repo) or results.configured_log_parser(data)
            else:
                parser = results.configured_log_parser(data) or MAP_REPO_TO_PARSER.get(repo)
            if parser is None:
                return {"resolved": False, "error": f"no_log_parser_for_{repo}"}
            test_content = eval_output
            if START_TEST_OUTPUT in eval_output and END_TEST_OUTPUT in eval_output:
                test_content = eval_output.split(START_TEST_OUTPUT, 1)[1].split(
                    END_TEST_OUTPUT,
                    1,
                )[0]
            status_map = parser(test_content, test_spec) or parser(eval_output, test_spec)
            if not status_map:
                return {"resolved": False, "error": "no_test_results_parsed"}

            eval_ref = {FAIL_TO_PASS: test_spec.FAIL_TO_PASS, PASS_TO_PASS: test_spec.PASS_TO_PASS}
            eval_type = EvalType.FAIL_ONLY if repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
            report = get_eval_tests_report(status_map, eval_ref, eval_type=eval_type)
            f2p_rate = compute_fail_to_pass(report)
            p2p_rate = compute_pass_to_pass(report)
            resolution = get_resolution_status(report)
            return {
                "resolved": resolution == ResolvedStatus.FULL.value,
                "resolution_status": resolution,
                "f2p_rate": f2p_rate,
                "p2p_rate": p2p_rate,
                "f2p_success": report[FAIL_TO_PASS]["success"],
                "f2p_failure": report[FAIL_TO_PASS]["failure"],
                "p2p_success": report[PASS_TO_PASS]["success"],
                "p2p_failure": report[PASS_TO_PASS]["failure"],
            }
        except (ImportError, KeyError, ValueError, TypeError, IndexError, AttributeError) as exc:
            return {"resolved": False, "error": f"grading_error: {exc}"}

    @staticmethod
    def _extract_env_exports(dockerfile_content: str) -> list[str]:
        exports: list[str] = []
        for line in dockerfile_content.splitlines():
            stripped = line.strip()
            if stripped.startswith("ENV "):
                exports.append("export " + stripped[4:])
        return exports
