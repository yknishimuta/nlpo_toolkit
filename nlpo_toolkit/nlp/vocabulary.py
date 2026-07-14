from __future__ import annotations

from pathlib import Path

__all__ = ["load_wordlist"]


def load_wordlist(path: Path) -> frozenset[str]:
    """Load an unnormalized UTF-8 wordlist containing one item per line."""
    return frozenset(
        word
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if (word := line.strip()) and not word.startswith("#")
    )
