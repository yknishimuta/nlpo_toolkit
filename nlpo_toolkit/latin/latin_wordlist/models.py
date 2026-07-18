from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from nlpo_toolkit.immutable_collections import freeze_count_mapping, freeze_mapping


@dataclass(frozen=True)
class WordlistFilterPolicy:
    min_length: int
    min_form_freq: int
    min_text_freq: int

    def __post_init__(self) -> None:
        for name, value in (
            ("min_length", self.min_length),
            ("min_form_freq", self.min_form_freq),
            ("min_text_freq", self.min_text_freq),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class WordlistTokenizationPolicy:
    extra_punct: str


@dataclass(frozen=True)
class LatinWordlistBuildRequest:
    config_path: Path
    conllu_dir: Path
    latin_text_dir: Path
    extra_wordlists: tuple[Path, ...]
    output_path: Path
    filters: WordlistFilterPolicy
    tokenization: WordlistTokenizationPolicy

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", self.config_path.resolve())
        object.__setattr__(self, "conllu_dir", self.conllu_dir.resolve())
        object.__setattr__(self, "latin_text_dir", self.latin_text_dir.resolve())
        object.__setattr__(
            self, "extra_wordlists", tuple(path.resolve() for path in self.extra_wordlists)
        )
        object.__setattr__(self, "output_path", self.output_path.resolve())


@dataclass(frozen=True)
class ConlluCandidates:
    files: tuple[Path, ...]
    lemmas: frozenset[str]
    form_counts: Mapping[str, int]
    ignored_rows: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "lemmas", frozenset(self.lemmas))
        object.__setattr__(self, "form_counts", freeze_count_mapping(self.form_counts))


@dataclass(frozen=True)
class TextCandidates:
    files: tuple[Path, ...]
    form_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "form_counts", freeze_count_mapping(self.form_counts))


@dataclass(frozen=True)
class ExtraWordlistCandidates:
    path: Path
    entries: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", frozenset(self.entries))


class WordlistNoticeCode(Enum):
    MISSING_CONLLU_DIRECTORY = "missing_conllu_directory"
    MISSING_TEXT_DIRECTORY = "missing_text_directory"
    MISSING_EXTRA_WORDLIST = "missing_extra_wordlist"


@dataclass(frozen=True)
class WordlistNotice:
    code: WordlistNoticeCode
    path: Path
    message: str


@dataclass(frozen=True)
class WordlistBuildStatistics:
    conllu_file_count: int
    conllu_lemma_count: int
    conllu_form_count: int
    text_file_count: int
    text_form_count: int
    extra_wordlist_counts: Mapping[Path, int]
    merged_word_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "extra_wordlist_counts", freeze_mapping(self.extra_wordlist_counts)
        )


@dataclass(frozen=True)
class WordlistPublication:
    output_path: Path
    entries: tuple[str, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if entries != tuple(sorted(set(entries))):
            raise ValueError("wordlist entries must be unique and sorted")
        object.__setattr__(self, "entries", entries)


@dataclass(frozen=True)
class LatinWordlistBuildResult:
    output_path: Path
    word_count: int
    statistics: WordlistBuildStatistics
    notices: tuple[WordlistNotice, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "notices", tuple(self.notices))
