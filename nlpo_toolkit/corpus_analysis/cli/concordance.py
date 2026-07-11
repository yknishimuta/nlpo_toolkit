from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..concordance import ConcordanceError, write_concordance
from .common import CLIContext, set_handler


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
        return write_concordance(
            tokens_path=args.tokens,
            keys=list(args.keys),
            field=args.field,
            window=args.window,
            output_format=args.format,
            out_path=args.out,
        )
    except ConcordanceError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
