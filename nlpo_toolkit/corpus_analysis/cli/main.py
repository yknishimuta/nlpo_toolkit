from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import cache, compare, concordance, count, features, ngram
from .common import CLIContext, CommandHandler


COMMAND_MODULES = (
    count,
    cache,
    concordance,
    compare,
    features,
    ngram,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nlpo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_module in COMMAND_MODULES:
        command_module.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv_list)
    handler: CommandHandler = args.handler
    return handler(args, CLIContext(argv=("nlpo", *argv_list)))
