from __future__ import annotations

from collections.abc import Iterator

__all__ = ["iter_char_chunks"]


def iter_char_chunks(text: str, chunk_chars: int) -> Iterator[str]:
    """Split text near the target size, preferring whitespace boundaries."""
    if (
        not isinstance(chunk_chars, int)
        or isinstance(chunk_chars, bool)
        or chunk_chars <= 0
    ):
        raise ValueError("chunk_chars must be a positive integer")
    start = 0
    while start < len(text):
        end = start + chunk_chars
        if end >= len(text):
            yield text[start:]
            break
        while end > start and not text[end].isspace():
            end -= 1
        if end == start:
            end = start + chunk_chars
        yield text[start:end]
        start = end
