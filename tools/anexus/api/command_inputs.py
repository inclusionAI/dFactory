"""Typed inputs accepted by the public command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class InitCommandInput:
    """Initialize one workspace."""

    workspace: str = ""
    force: bool = False


@dataclass(frozen=True, kw_only=True)
class UpdateCommandInput:
    """Synchronize the catalog configured by one workspace."""

    workspace: str = ""


@dataclass(frozen=True, kw_only=True)
class RunCommandInput:
    """Select tasks and run-scoped capacity for planning or execution."""

    workspace: str = ""
    tasks: str
    run_id: str = ""
    task_ids: tuple[str, ...] = ()
    attempts: int = 1
    concurrency: int = 10


@dataclass(frozen=True, kw_only=True)
class InspectCommandInput:
    """Select one durable run for inspection."""

    workspace: str = ""
    run_id: str


CommandInput = InitCommandInput | UpdateCommandInput | RunCommandInput | InspectCommandInput


def command_input_from_namespace(args: argparse.Namespace) -> CommandInput:
    """Convert the parser boundary into one closed command input type."""

    workspace = str(args.workspace or "")
    if args.command == "init":
        return InitCommandInput(workspace=workspace, force=bool(args.force))
    if args.command == "update":
        return UpdateCommandInput(workspace=workspace)
    if args.command in {"plan", "run"}:
        return RunCommandInput(
            workspace=workspace,
            tasks=str(args.tasks),
            run_id=str(args.run_id or ""),
            task_ids=tuple(args.task_id),
            attempts=int(args.attempts),
            concurrency=int(args.concurrency),
        )
    if args.command == "inspect":
        return InspectCommandInput(workspace=workspace, run_id=str(args.run_id))
    raise ValueError(f"unsupported command: {args.command}")
