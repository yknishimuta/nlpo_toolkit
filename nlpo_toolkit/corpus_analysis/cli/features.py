from __future__ import annotations

import argparse
from pathlib import Path

from ..composition import default_feature_command_dependencies
from ..corpus_errors import CorpusPreparationError
from ..features.errors import FeatureError
from ..features.models import (
    FeatureRequest,
    FeatureSamplingOptions,
    FunctionWordSource,
    LexicalDiversityOptions,
)
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
        "--window-tokens",
        type=int,
        default=None,
        help="Emit one feature row per fixed-size window of eligible word tokens.",
    )
    parser.add_argument(
        "--step-tokens",
        type=int,
        default=None,
        help="Window start interval. Requires --window-tokens; defaults to its value.",
    )
    parser.add_argument(
        "--include-partial-window",
        action="store_true",
        help="Include at most one trailing window shorter than --window-tokens.",
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
        "--lexical-diversity",
        action="store_true",
        help="Include MATTR, MSTTR, MTLD, and HD-D for tokens and lemmas.",
    )
    parser.add_argument(
        "--lexdiv-window",
        type=int,
        default=None,
        help="MATTR/MSTTR window size. Implies --lexical-diversity; default: 100.",
    )
    parser.add_argument(
        "--mtld-threshold",
        type=float,
        default=None,
        help="MTLD factor threshold. Implies --lexical-diversity; default: 0.72.",
    )
    parser.add_argument(
        "--hdd-sample-size",
        type=int,
        default=None,
        help="HD-D sample size. Implies --lexical-diversity; default: 42.",
    )
    parser.add_argument(
        "--function-words",
        type=Path,
        default=None,
        help="UTF-8 file containing one explicit function word per line.",
    )
    parser.add_argument(
        "--function-word-field",
        choices=("lemma", "token"),
        default=None,
        help="Field used for explicit function-word matching; default: lemma.",
    )
    add_grouping_override_arguments(parser)
    add_empty_group_argument(parser)
    set_handler(parser, execute)


def execute(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        if args.function_word_field is not None and args.function_words is None:
            raise FeatureError("--function-word-field requires --function-words")
        lexical_diversity = None
        if args.lexical_diversity or any(
            value is not None
            for value in (
                args.lexdiv_window,
                args.mtld_threshold,
                args.hdd_sample_size,
            )
        ):
            lexical_diversity = LexicalDiversityOptions(
                window_size=(
                    args.lexdiv_window if args.lexdiv_window is not None else 100
                ),
                mtld_threshold=(
                    args.mtld_threshold if args.mtld_threshold is not None else 0.72
                ),
                hdd_sample_size=(
                    args.hdd_sample_size if args.hdd_sample_size is not None else 42
                ),
            )
        result = execute_feature_command(
            FeatureRequest(
                corpus=build_corpus_preparation_request(args),
                field=args.field,
                mfw=args.mfw,
                include_upos=bool(args.include_upos),
                include_basic=bool(args.include_basic),
                sampling=FeatureSamplingOptions(
                    window_tokens=args.window_tokens,
                    step_tokens=args.step_tokens,
                    include_partial=args.include_partial_window,
                ),
                lexical_diversity=lexical_diversity,
                function_words=(
                    FunctionWordSource(
                        path=args.function_words,
                        field=args.function_word_field or "lemma",
                    )
                    if args.function_words is not None
                    else None
                ),
            ),
            dependencies=default_feature_command_dependencies(),
        )
        with open_cli_output(path=args.out, stdout=context.stdout) as stream:
            write_feature_result(result, stream=stream, output_format=args.format)
        return 0
    except (CorpusPreparationError, FeatureError, ValueError, FileNotFoundError) as exc:
        present_error(exc, stderr=context.stderr)
        return 1
