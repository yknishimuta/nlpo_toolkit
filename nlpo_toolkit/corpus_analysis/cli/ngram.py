from __future__ import annotations

import argparse
from pathlib import Path

from ..cleaner_runtime import CleanerError
from ..dependencies import default_config_ngram_dependencies
from ..ngram import (
    ConfigNgramRequest,
    NgramError,
    TokenNgramRequest,
    execute_config_ngram_command,
    execute_token_ngram_command,
)
from .common import CLIContext, resolve_config_path, resolve_project_root, set_handler
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
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root used with --config input.",
    )
    inputs.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config path for groups input. Defaults to <project-root>/config/groups.config.yml.",
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
    parser.add_argument(
        "--group-by-file",
        action="store_true",
        help="Process each configured input file as a separate n-gram group.",
    )
    parser.add_argument(
        "--auto-single-cleaned",
        action="store_true",
        help="Use the only .txt file in cleaned_dir; fail if zero or multiple files exist.",
    )
    parser.add_argument(
        "--error-on-empty-group",
        action="store_true",
        help="Fail when any configured group matches zero files.",
    )
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
            project_root = resolve_project_root(args.project_root)
            config_path = resolve_config_path(
                project_root=project_root, config_path=args.config
            )
            result = execute_config_ngram_command(
                request=ConfigNgramRequest(
                    project_root=project_root,
                    config_path=config_path,
                    n=args.n,
                    field=args.field,
                    by_group=bool(args.by_group),
                    min_count=args.min_count,
                    top=args.top,
                    group_by_file=bool(args.group_by_file),
                    auto_single_cleaned=bool(args.auto_single_cleaned),
                    error_on_empty_group=bool(args.error_on_empty_group),
                ),
                dependencies=default_config_ngram_dependencies(),
            )
        with open_cli_output(path=args.out, stdout=context.stdout) as stream:
            write_ngram_result(result, stream=stream, output_format=args.format)
        return 0
    except (CleanerError, NgramError, ValueError, FileNotFoundError) as exc:
        present_error(exc, stderr=context.stderr)
        return 1
