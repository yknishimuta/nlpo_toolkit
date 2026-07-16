from __future__ import annotations

import argparse
from pathlib import Path

from ..composition import default_config_ngram_dependencies
from ..corpus_errors import CorpusPreparationError
from ..ngram import (
    ConfigNgramRequest,
    NgramError,
    TokenNgramRequest,
    execute_config_ngram_command,
    execute_token_ngram_command,
)
from .common import (
    CLIContext,
    add_config_argument,
    add_empty_group_argument,
    add_grouping_override_arguments,
    add_project_root_argument,
    build_corpus_preparation_request,
    set_handler,
)
from .output import open_cli_output, present_error, write_ngram_result


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("ngram")
    parser.add_argument("--n", type=int, default=2, help="N-gram size.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--tokens",
        type=Path,
        default=None,
        help="Complete token artifact TSV path.",
    )
    add_project_root_argument(
        parser,
        help_text="Project root used with --config input.",
    )
    add_config_argument(
        inputs,
        help_text=(
            "YAML config path for groups input. Defaults to "
            "<project-root>/config/groups.config.yml."
        ),
    )
    parser.add_argument(
        "--field",
        choices=("token", "lemma"),
        default="lemma",
        help="Token artifact field to use for n-grams.",
    )
    parser.add_argument(
        "--by-group",
        action="store_true",
        help="Aggregate n-grams separately for each token artifact group.",
    )
    parser.add_argument("--min-count", type=int, default=1, help="Minimum frequency to include.")
    parser.add_argument("--top", type=int, default=None, help="Limit output to the top N n-grams.")
    parser.add_argument("--format", choices=("tsv", "csv"), default="tsv", help="Output format.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path. Defaults to standard output.",
    )
    add_grouping_override_arguments(parser)
    add_empty_group_argument(parser)
    set_handler(parser, execute)


def execute(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        if args.tokens is not None:
            result = execute_token_ngram_command(
                TokenNgramRequest(
                    tokens_path=args.tokens,
                    n=args.n,
                    field=args.field,
                    by_group=bool(args.by_group),
                    min_count=args.min_count,
                    top=args.top,
                )
            )
        else:
            result = execute_config_ngram_command(
                request=ConfigNgramRequest(
                    corpus=build_corpus_preparation_request(args),
                    n=args.n,
                    by_group=bool(args.by_group),
                    min_count=args.min_count,
                    top=args.top,
                ),
                dependencies=default_config_ngram_dependencies(),
            )
        with open_cli_output(path=args.out, stdout=context.stdout) as stream:
            write_ngram_result(result, stream=stream, output_format=args.format)
        return 0
    except (CorpusPreparationError, NgramError, ValueError, FileNotFoundError) as exc:
        present_error(exc, stderr=context.stderr)
        return 1
