from __future__ import annotations

import argparse
from pathlib import Path

from nlpo_toolkit.comparison.cli_service import (
    CompareError,
    CompareRequest,
    execute_compare_command,
)

from .common import CLIContext, set_handler
from .output import open_cli_output, present_error, write_compare_result


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("compare")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Two or more frequency CSV files to compare.",
    )
    parser.add_argument("--labels", nargs="+", default=None, help="Display labels corresponding to --inputs.")
    parser.add_argument("--out", type=Path, default=None, help="Output path. Defaults to standard output.")
    parser.add_argument("--format", choices=("csv", "tsv"), default="csv", help="Output format.")
    parser.add_argument(
        "--metric",
        choices=("relative", "difference", "ratio", "log-ratio"),
        default="log-ratio",
        help="Primary comparison metric. Rows include all available comparison columns.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help="Additive smoothing used for ratio and log-ratio calculations.",
    )
    parser.add_argument(
        "--sort",
        choices=("abs-log-ratio", "log-ratio", "difference", "range-relative", "total", "term"),
        default=None,
        help="Sort key. Defaults to abs-log-ratio for two inputs and range-relative for three or more.",
    )
    parser.add_argument("--ascending", action="store_true", help="Sort ascending.")
    parser.add_argument("--descending", action="store_true", help="Sort descending. This is the default.")
    parser.add_argument("--top", type=int, default=None, help="Limit output to the top N rows after sorting.")
    parser.add_argument(
        "--min-total-count",
        type=float,
        default=1,
        help="Exclude terms whose summed count across inputs is below this value.",
    )
    parser.add_argument(
        "--key-column",
        default=None,
        help="Explicit key column name. Defaults to automatic detection.",
    )
    parser.add_argument(
        "--count-column",
        default=None,
        help="Explicit count column name. Defaults to automatic detection.",
    )
    set_handler(parser, execute)


def execute(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        result = execute_compare_command(
            CompareRequest(
            inputs=tuple(args.inputs),
            labels=tuple(args.labels) if args.labels is not None else None,
            metric=args.metric,
            smoothing=args.smoothing,
            min_total_count=args.min_total_count,
            top=args.top,
            sort=args.sort,
            ascending=bool(args.ascending) and not bool(args.descending),
            key_column=args.key_column,
            count_column=args.count_column,
            )
        )
        with open_cli_output(path=args.out, stdout=context.stdout) as stream:
            write_compare_result(result, stream=stream, output_format=args.format)
        return 0
    except CompareError as exc:
        present_error(exc, stderr=context.stderr)
        return 1
