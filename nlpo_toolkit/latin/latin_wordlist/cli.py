from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .composition import default_latin_wordlist_dependencies
from .config import load_wordlist_build_request
from .errors import LatinWordlistConfigError, LatinWordlistError
from .service import execute_latin_wordlist_build


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "latin_wordlist.yml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a general-purpose Latin wordlist")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the strict YAML configuration file",
    )
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        request = load_wordlist_build_request(args.config)
    except LatinWordlistConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    try:
        result = execute_latin_wordlist_build(
            request, dependencies=default_latin_wordlist_dependencies()
        )
    except LatinWordlistError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    for notice in result.notices:
        print(f"[WARN] {notice.message}", file=sys.stderr)
    statistics = result.statistics
    print(f"[INFO] lemmas from conllu : {statistics.conllu_lemma_count:,}")
    print(f"[INFO] forms from conllu  : {statistics.conllu_form_count:,}")
    print(f"[INFO] forms from txt     : {statistics.text_form_count:,}")
    for path, count in statistics.extra_wordlist_counts.items():
        print(f"[INFO] extra wordlist {path}: {count:,}")
    print(f"[OK] wrote {result.word_count:,} words -> {result.output_path}")
    return 0


def main() -> int:
    return run_cli()
