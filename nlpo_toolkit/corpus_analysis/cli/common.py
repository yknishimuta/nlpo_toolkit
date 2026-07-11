from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class CommandHandler(Protocol):
    def __call__(self, args: argparse.Namespace, context: "CLIContext") -> int:
        ...


@dataclass(frozen=True)
class CLIContext:
    argv: tuple[str, ...]


def resolve_project_root(path: Path) -> Path:
    return path.resolve()


def resolve_config_path(
    *,
    project_root: Path,
    config_path: Path | None,
) -> Path:
    if config_path is None:
        return project_root / "config" / "groups.config.yml"
    if config_path.is_absolute():
        return config_path
    return (project_root / config_path).resolve()


def add_project_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    project_root_help: str,
    config_help: str,
) -> None:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help=project_root_help,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=config_help,
    )


def set_handler(parser: argparse.ArgumentParser, handler: CommandHandler) -> None:
    parser.set_defaults(handler=handler)
