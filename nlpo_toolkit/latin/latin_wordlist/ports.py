from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import (
    ConlluCandidates,
    ExtraWordlistCandidates,
    TextCandidates,
    WordlistNotice,
    WordlistPublication,
    WordlistTokenizationPolicy,
)


class ConlluCandidateCollector(Protocol):
    def __call__(self, *, directory: Path, min_length: int) -> tuple[
        ConlluCandidates, tuple[WordlistNotice, ...]
    ]: ...


class TextCandidateCollector(Protocol):
    def __call__(
        self,
        *,
        directory: Path,
        policy: WordlistTokenizationPolicy,
        min_length: int,
    ) -> tuple[TextCandidates, tuple[WordlistNotice, ...]]: ...


class ExtraWordlistCollector(Protocol):
    def __call__(self, *, path: Path) -> tuple[
        ExtraWordlistCandidates | None, tuple[WordlistNotice, ...]
    ]: ...


class WordlistPublisher(Protocol):
    def __call__(self, publication: WordlistPublication) -> None: ...


@dataclass(frozen=True)
class LatinWordlistDependencies:
    collect_conllu: ConlluCandidateCollector
    collect_text: TextCandidateCollector
    collect_extra_wordlist: ExtraWordlistCollector
    publish: WordlistPublisher
