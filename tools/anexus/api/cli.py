"""Command-line entrypoint."""

from __future__ import annotations

import sys
from collections.abc import Callable

from anexus.api import config_commands, run_command
from anexus.api.arguments import build_parser
from anexus.api.command_inputs import CommandInput, command_input_from_namespace
from anexus.api.inspection_command import inspect_command

Command = Callable[[CommandInput], int]

_COMMANDS: dict[str, Command] = {
    "init": config_commands.init_command,
    "update": config_commands.update_command,
    "plan": config_commands.plan_command,
    "run": run_command.run_command,
    "inspect": inspect_command,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = _COMMANDS.get(args.command)
    assert command is not None
    try:
        return command(command_input_from_namespace(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
