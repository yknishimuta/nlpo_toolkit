"""Deprecated; use nlpo_toolkit.corpus_analysis.cli."""

from nlpo_toolkit.corpus_analysis.cli import *
from nlpo_toolkit.corpus_analysis.cli import build_parser, main

__all__ = ["build_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
