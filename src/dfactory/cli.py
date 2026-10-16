"""Command-line interface for dFactory."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from dfactory.config import ConfigError, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dfactory", description="Train and extend discrete diffusion language models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run a training recipe.")
    train_parser.add_argument("--config", required=True, help="Path to a YAML run config.")
    train_parser.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE", help="Override a config value."
    )

    config_parser = subparsers.add_parser("config", help="Inspect run configuration.")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    validate_parser = config_subparsers.add_parser("validate", help="Validate and resolve a YAML config.")
    validate_parser.add_argument("--config", required=True, help="Path to a YAML run config.")
    validate_parser.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE", help="Override a config value."
    )

    recipes_parser = subparsers.add_parser("recipes", help="Inspect available recipes.")
    recipes_subparsers = recipes_parser.add_subparsers(dest="recipes_command", required=True)
    recipes_subparsers.add_parser("list", help="List recipe names.")
    describe_parser = recipes_subparsers.add_parser("describe", help="Describe one recipe.")
    describe_parser.add_argument("name")

    datasets_parser = subparsers.add_parser("datasets", help="Inspect available dataset sources.")
    datasets_subparsers = datasets_parser.add_subparsers(dest="datasets_command", required=True)
    datasets_subparsers.add_parser("list", help="List dataset-source names.")

    models_parser = subparsers.add_parser("models", help="Inspect available model providers.")
    models_subparsers = models_parser.add_subparsers(dest="models_command", required=True)
    models_subparsers.add_parser("list", help="List model-provider names.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            from dfactory.api import train

            train(args.config, overrides=args.set)
            return 0

        if args.command == "config" and args.config_command == "validate":
            config = load_config(args.config, overrides=args.set)
            print(config.to_yaml(), end="")
            return 0

        if args.command == "recipes":
            from dfactory.recipes import describe_recipe, list_recipes

            if args.recipes_command == "list":
                for name in list_recipes():
                    print(name)
            else:
                spec = describe_recipe(args.name)
                print(f"{spec.name}: {spec.description}")
            return 0

        if args.command == "datasets" and args.datasets_command == "list":
            from dfactory.data import list_datasets

            for name in list_datasets():
                print(name)
            return 0

        if args.command == "models" and args.models_command == "list":
            from dfactory.models import list_models

            for name in list_models():
                print(name)
            return 0
    except (ConfigError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    parser.error("unsupported command")
    return 2


__all__ = ["main"]
