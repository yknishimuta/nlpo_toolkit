from __future__ import annotations

import argparse
from pathlib import Path

from ..concordance import ConcordanceError, ConcordanceRequest, build_concordance
from .common import CLIContext, set_handler
from .output import open_cli_output, present_error, write_concordance_result


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("concordance")
    parser.add_argument(
        "--tokens",
        type=Path,
        required=True,
        help="Complete token artifact TSV path.",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        required=True,
        help="Search keys. Multiple values are accepted.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Number of words to show on each side of the matched token.",
    )
    parser.add_argument(
        "--field",
        choices=("token", "lemma"),
        default="lemma",
        help="Token artifact field to search.",
    )
    parser.add_argument("--format", choices=("tsv", "csv"), default="tsv", help="Output format.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path. Defaults to standard output.",
    )
    set_handler(parser, execute)


def execute(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        result = build_concordance(
            ConcordanceRequest(
                tokens_path=args.tokens,
                keys=tuple(args.keys),
                field=args.field,
                window=args.window,
            )
        )
        with open_cli_output(path=args.out, stdout=context.stdout) as stream:
            write_concordance_result(result, stream=stream, output_format=args.format)
        return 0
    except ConcordanceError as exc:
        present_error(exc, stderr=context.stderr)
        return 1
