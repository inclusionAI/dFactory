"""SWE-bench test-log parsing and result reduction."""

from __future__ import annotations

import importlib
from typing import Any

from anexus.tasks.graders.swebench_payload import parse_list_field


def configured_log_parser(data: dict[str, Any]):
    parser_name = ""
    install_config = data.get("install_config")
    if isinstance(install_config, dict):
        parser_name = str(install_config.get("log_parser", "") or "").strip()
    if not parser_name:
        parser_name = str(data.get("log_parser", "") or "").strip()
    if not parser_name:
        return None

    for module_name in (
        "swebench.harness.log_parsers",
        "swebench.harness.log_parsers.python",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        parser = getattr(module, parser_name, None)
        if callable(parser):
            return parser
    return None


def grade_status_map(status_map: dict[str, str], data: dict[str, Any]) -> dict[str, Any]:
    fail_to_pass = parse_list_field(data, "FAIL_TO_PASS")
    pass_to_pass = parse_list_field(data, "PASS_TO_PASS")
    f2p_success, f2p_failure = _bucket_tests(status_map, fail_to_pass)
    p2p_success, p2p_failure = _bucket_tests(status_map, pass_to_pass)
    f2p_rate = len(f2p_success) / len(fail_to_pass) if fail_to_pass else 1.0
    p2p_rate = len(p2p_success) / len(pass_to_pass) if pass_to_pass else 1.0
    resolved = f2p_rate == 1.0 and p2p_rate == 1.0
    return {
        "resolved": resolved,
        "resolution_status": "RESOLVED_FULL" if resolved else "RESOLVED_NO",
        "f2p_rate": f2p_rate,
        "p2p_rate": p2p_rate,
        "f2p_success": f2p_success,
        "f2p_failure": f2p_failure,
        "p2p_success": p2p_success,
        "p2p_failure": p2p_failure,
    }


def _bucket_tests(status_map: dict[str, str], names: list[str]) -> tuple[list[str], list[str]]:
    success: list[str] = []
    failure: list[str] = []
    for name in names:
        if status_map.get(name) == "PASSED":
            success.append(name)
        else:
            failure.append(name)
    return success, failure
