from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

from nlpo_toolkit.nlp.chunking import iter_char_chunks


ChunkStrategy = Literal["char_whitespace"]


@dataclass(frozen=True)
class AnalysisExtractionPolicy:
    chunk_chars: int = 200_000
    chunk_strategy: ChunkStrategy = "char_whitespace"
    processors: tuple[str, ...] = ("tokenize", "mwt", "pos", "lemma")

    def __post_init__(self) -> None:
        if (
            not isinstance(self.chunk_chars, int)
            or isinstance(self.chunk_chars, bool)
            or self.chunk_chars <= 0
        ):
            raise ValueError("chunk_chars must be a positive integer")
        if self.chunk_strategy != "char_whitespace":
            raise ValueError(f"Unsupported chunk strategy: {self.chunk_strategy}")
        if not self.processors:
            raise ValueError("processors must not be empty")
        normalized = tuple(processor.strip() for processor in self.processors)
        if any(not processor for processor in normalized):
            raise ValueError("processors must not contain empty names")
        if len(set(normalized)) != len(normalized):
            raise ValueError("processors must be unique")
        object.__setattr__(self, "processors", normalized)


DEFAULT_ANALYSIS_EXTRACTION_POLICY = AnalysisExtractionPolicy()


def iter_analysis_chunks(
    text: str,
    *,
    policy: AnalysisExtractionPolicy,
) -> Iterator[str]:
    if policy.chunk_strategy == "char_whitespace":
        yield from iter_char_chunks(text, chunk_chars=policy.chunk_chars)
        return
    raise ValueError(f"Unsupported chunk strategy: {policy.chunk_strategy}")
