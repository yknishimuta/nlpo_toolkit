from __future__ import annotations

import argparse
from pathlib import Path

from ..composition import default_feature_command_dependencies
from ..corpus_errors import CorpusPreparationError
from ..features.errors import FeatureError
from ..features.models import FeatureRequest
from ..features.service import execute_feature_command
from .common import (
    CLIContext,
    add_empty_group_argument,
    add_grouping_override_arguments,
    add_project_config_arguments,
    build_corpus_preparation_request,
    set_handler,
)
from .output import open_cli_output, present_error, write_feature_result


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
    add_grouping_override_arguments(parser)
    add_empty_group_argument(parser)
    set_handler(parser, execute)


def execute(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        result = execute_feature_command(
            FeatureRequest(
                corpus=build_corpus_preparation_request(args),
                field=args.field,
                mfw=args.mfw,
                include_upos=bool(args.include_upos),
                include_basic=bool(args.include_basic),
            ),
            dependencies=default_feature_command_dependencies(),
        )
        with open_cli_output(path=args.out, stdout=context.stdout) as stream:
            write_feature_result(result, stream=stream, output_format=args.format)
        return 0
    except (CorpusPreparationError, FeatureError, ValueError, FileNotFoundError) as exc:
        present_error(exc, stderr=context.stderr)
        return 1
