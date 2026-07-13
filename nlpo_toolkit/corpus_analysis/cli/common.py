from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from ..requests import CorpusPreparationRequest, GroupingOverride


class CommandHandler(Protocol):
    def __call__(self, args: argparse.Namespace, context: "CLIContext") -> int:
        ...


@dataclass(frozen=True)
class CLIContext:
    argv: tuple[str, ...]
    stdout: TextIO
    stderr: TextIO


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
        return config_path.resolve()
    return (project_root / config_path).resolve()


def add_project_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    project_root_help: str,
    config_help: str,
) -> None:
    add_project_root_argument(parser, help_text=project_root_help)
    add_config_argument(parser, help_text=config_help)


def add_project_root_argument(
    parser: argparse._ActionsContainer,
    *,
    help_text: str,
) -> None:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help=help_text,
    )


def add_config_argument(
    parser: argparse._ActionsContainer,
    *,
    help_text: str,
) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=help_text,
    )


def add_grouping_override_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--group-by-file",
        action="store_true",
        help="Process each configured input file separately.",
    )
    group.add_argument(
        "--auto-single-cleaned",
        action="store_true",
        help="Use the only .txt file in cleaned_dir; fail unless exactly one exists.",
    )


def add_empty_group_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--error-on-empty-group",
        action="store_true",
        help="Fail when any configured group matches zero files.",
    )


def build_corpus_preparation_request(
    args: argparse.Namespace,
) -> CorpusPreparationRequest:
    project_root = resolve_project_root(args.project_root)
    config_path = resolve_config_path(
        project_root=project_root,
        config_path=args.config,
    )
    if args.group_by_file and args.auto_single_cleaned:
        raise ValueError("grouping overrides are mutually exclusive")
    grouping_override: GroupingOverride | None = None
    if args.group_by_file:
        grouping_override = "per_file"
    elif args.auto_single_cleaned:
        grouping_override = "auto_single_cleaned"
    return CorpusPreparationRequest(
        project_root=project_root,
        config_path=config_path,
        grouping_override=grouping_override,
        error_on_empty_group=bool(args.error_on_empty_group),
    )


def set_handler(parser: argparse.ArgumentParser, handler: CommandHandler) -> None:
    parser.set_defaults(handler=handler)
