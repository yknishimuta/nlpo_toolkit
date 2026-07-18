from __future__ import annotations

import argparse
from pathlib import Path

from ..composition import default_feature_command_dependencies
from ..corpus_errors import CorpusPreparationError
from ..features.errors import FeatureError
from ..features.service import execute_feature_command
from .common import (
    CLIContext,
    add_empty_group_argument,
    add_grouping_override_arguments,
    add_project_config_arguments,
    build_corpus_preparation_request,
    set_handler,
)
from .feature_options import add_feature_options, build_feature_request
from .output import open_cli_output, present_error, write_feature_result


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("features")
    add_project_config_arguments(
        parser,
        project_root_help="Project root used to resolve relative paths in the config.",
        config_help="YAML config path. Defaults to <project-root>/config/groups.config.yml.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--format", choices=("csv", "tsv"), default="csv")
    add_feature_options(parser)
    add_grouping_override_arguments(parser)
    add_empty_group_argument(parser)
    set_handler(parser, execute)


def execute(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        request = build_feature_request(
            args, corpus=build_corpus_preparation_request(args)
        )
        result = execute_feature_command(
            request, dependencies=default_feature_command_dependencies()
        )
        with open_cli_output(path=args.out, stdout=context.stdout) as stream:
            write_feature_result(result, stream=stream, output_format=args.format)
        return 0
    except (CorpusPreparationError, FeatureError, ValueError, FileNotFoundError) as exc:
        present_error(exc, stderr=context.stderr)
        return 1
