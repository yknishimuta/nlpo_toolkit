from __future__ import annotations

from pathlib import Path


class CorpusPreparationError(ValueError):
    """Corpus preparation could not complete."""


class CorpusReadError(CorpusPreparationError):
    """A configured corpus file could not be read as UTF-8."""

    def __init__(self, path: Path, message: str) -> None:
        self.path = Path(path)
        super().__init__(
            f"Failed to read corpus file as UTF-8: {self.path}: {message}"
        )
