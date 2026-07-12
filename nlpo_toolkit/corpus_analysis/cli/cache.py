from __future__ import annotations

import argparse
from pathlib import Path

from ..cache import CacheClearError, CacheClearRequest, clear_cache
from .common import CLIContext, resolve_project_root, set_handler
from .output import present_error


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
        result = clear_cache(
            CacheClearRequest(project_root=project_root, config_path=config_path)
        )
        display = result.cache_dir.relative_to(project_root).as_posix()
        message = "cache cleared" if result.removed else "cache already clean"
        print(f"[OK] {message}: {display}", file=context.stdout)
        return 0
    except CacheClearError as exc:
        present_error(exc, stderr=context.stderr)
        return 1
