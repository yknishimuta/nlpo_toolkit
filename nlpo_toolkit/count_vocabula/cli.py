"""Deprecated; use nlpo_toolkit.corpus_analysis.cli."""

from nlpo_toolkit.corpus_analysis.cli import *
from nlpo_toolkit.corpus_analysis.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
