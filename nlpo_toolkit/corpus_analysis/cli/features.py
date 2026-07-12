from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..cleaner_runtime import CleanerError
from ..dependencies import default_feature_command_dependencies
from ..features import FeatureError, FeatureRequest, execute_feature_command
from .common import (
    CLIContext,
    add_project_config_arguments,
    resolve_config_path,
    resolve_project_root,
    set_handler,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("features")
    add_project_config_arguments(
        parser,
        project_root_help="Project root used to resolve relative paths in the config.",
        config_help="YAML config path. Defaults to <project-root>/config/groups.config.yml.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV/TSV path. Defaults to standard output.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "tsv"),
        default="csv",
        help="Output format.",
    )
    parser.add_argument(
        "--field",
        choices=("lemma", "token"),
        default="lemma",
        help="Field used for MFW features.",
    )
    parser.add_argument(
        "--mfw",
        type=int,
        default=0,
        help="Add relative-frequency features for the top N most frequent words/lemmas.",
    )
    parser.add_argument(
        "--include-upos",
        dest="include_upos",
        action="store_true",
        default=True,
        help="Include UPOS count and ratio features. Enabled by default.",
    )
    parser.add_argument(
        "--no-upos",
        dest="include_upos",
        action="store_false",
        help="Disable UPOS features.",
    )
    parser.add_argument(
        "--include-basic",
        dest="include_basic",
        action="store_true",
        default=True,
        help="Include basic text statistics. Enabled by default.",
    )
    parser.add_argument(
        "--no-basic",
        dest="include_basic",
        action="store_false",
        help="Disable basic text statistics.",
    )
    parser.add_argument(
        "--group-by-file",
        action="store_true",
        help="Write one feature row per input file instead of one row per configured group.",
    )
    parser.add_argument(
        "--auto-single-cleaned",
        action="store_true",
        help="Use the only .txt file in cleaned_dir as the feature target; fail if zero or multiple files exist.",
    )
    parser.add_argument(
        "--error-on-empty-group",
        action="store_true",
        help="Fail when any configured group matches zero files.",
    )
    set_handler(parser, execute)


def execute(args: argparse.Namespace, context: CLIContext) -> int:
    project_root = resolve_project_root(args.project_root)
    config_path = resolve_config_path(project_root=project_root, config_path=args.config)
    try:
        return execute_feature_command(
            FeatureRequest(
                project_root=project_root,
                config_path=config_path,
                out_path=args.out,
                output_format=args.format,
                field=args.field,
                mfw=args.mfw,
                include_upos=bool(args.include_upos),
                include_basic=bool(args.include_basic),
                group_by_file=bool(args.group_by_file),
                auto_single_cleaned=bool(args.auto_single_cleaned),
                error_on_empty_group=bool(args.error_on_empty_group),
            ),
            dependencies=default_feature_command_dependencies(),
        )
    except (CleanerError, FeatureError, ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
