from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..cache import CacheClearError, clear_cache
from .common import CLIContext, resolve_project_root, set_handler


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("cache")
    cache_subparsers = parser.add_subparsers(dest="cache_command", required=True)
    clear_parser = cache_subparsers.add_parser("clear")
    clear_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root used to resolve the configured cache directory.",
    )
    clear_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path. Defaults to <project-root>/config/groups.config.yml when it exists.",
    )
    set_handler(clear_parser, execute_clear)


def execute_clear(args: argparse.Namespace, context: CLIContext) -> int:
    project_root = resolve_project_root(args.project_root)
    config_path = args.config
    if config_path is not None and not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    try:
        return clear_cache(project_root=project_root, config_path=config_path)
    except CacheClearError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
