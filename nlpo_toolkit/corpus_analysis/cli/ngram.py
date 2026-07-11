from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from ..ngram import NgramError, write_ngrams_from_config, write_ngrams_from_tokens
from .common import CLIContext, resolve_config_path, resolve_project_root, set_handler


try:
    from nlpo_toolkit.latin.cleaners import run_clean_corpus as clean_mod
except Exception:
    clean_mod = SimpleNamespace(main=lambda argv: 0)


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
    set_handler(parser, execute)


def execute(args: argparse.Namespace, context: CLIContext) -> int:
    try:
        if args.tokens is not None:
            return write_ngrams_from_tokens(
                tokens_path=args.tokens,
                n=args.n,
                field=args.field,
                by_group=bool(args.by_group),
                min_count=args.min_count,
                top=args.top,
                output_format=args.format,
                out_path=args.out,
            )

        project_root = resolve_project_root(args.project_root)
        config_path = resolve_config_path(project_root=project_root, config_path=args.config)
        return write_ngrams_from_config(
            project_root=project_root,
            config_path=config_path,
            n=args.n,
            field=args.field,
            by_group=bool(args.by_group),
            min_count=args.min_count,
            top=args.top,
            output_format=args.format,
            out_path=args.out,
            clean_mod=clean_mod,
        )
    except NgramError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
