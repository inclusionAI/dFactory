"""ANexus command-line argument declarations."""

from __future__ import annotations

import argparse

PUBLIC_COMMANDS = ("init", "update", "plan", "run", "inspect")


def build_parser() -> argparse.ArgumentParser:
    """Build the deliberately small public CLI surface."""

    parser = argparse.ArgumentParser(prog="anexus")
    parser.add_argument(
        "--workspace",
        default="",
        help="Workspace root; defaults to ANEXUS_WORKSPACE or the current directory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a local workspace")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing files")

    subparsers.add_parser("update", help="Update the configured catalog checkout")

    plan_parser = subparsers.add_parser("plan", help="Resolve and validate a run plan")
    add_run_selection_args(plan_parser)

    run_parser = subparsers.add_parser("run", help="Run selected tasks")
    add_run_selection_args(run_parser)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a local durable run")
    inspect_parser.add_argument("run_id", help="Run id")
    return parser


def add_run_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add only caller-owned task selection and run-capacity inputs."""

    parser.add_argument("tasks", help="Workspace-relative local task JSONL file")
    parser.add_argument("--run-id", default="", help="Run id; generated when omitted")
    parser.add_argument("--task-id", action="append", default=[], help="Task id filter")
    parser.add_argument("--attempts", type=int, default=1, help="Attempts per selected task")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent task attempts")
