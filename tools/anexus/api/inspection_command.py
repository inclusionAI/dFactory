"""Thin CLI adapter for durable run inspection."""

from __future__ import annotations

import json

from anexus.api.cli_context import workspace_from_input
from anexus.api.command_inputs import InspectCommandInput
from anexus.api.inspection import build_run_inspection
from anexus.runtime import LocalRunStore


def inspect_command(command: InspectCommandInput) -> int:
    """Print one run inspection from the workspace's canonical output root."""

    store = LocalRunStore(workspace_from_input(command).cache_dir)
    print(
        json.dumps(
            build_run_inspection(store, run_id=command.run_id),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0
