from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol


CleanerKind = Literal["corpus_corporum", "scholastic_text"]
CLEANER_KINDS: frozenset[str] = frozenset({"corpus_corporum", "scholastic_text"})


class CleanerRunner(Protocol):
    def main(self, argv: list[str] | None = None) -> int | None: ...


CleanerLoader = Callable[[], CleanerRunner]


class CleanerConfigError(ValueError):
    """A cleaner configuration could not be inspected."""


class CleanerConfigReadError(CleanerConfigError):
    pass


class CleanerConfigParseError(CleanerConfigError):
    pass


class CleanerConfigValidationError(CleanerConfigError):
    pass


@dataclass(frozen=True)
class CleanerConfig:
    source_path: Path
    kind: CleanerKind
    input_path: Path
    output_path: Path
    rules_path: Path | None = None
    lexicon_map_path: Path | None = None
    ref_tsv_path: Path | None = None
    output_filename_template: str | None = None
    doc_id_prefix: str | None = None


@dataclass(frozen=True)
class CleanerReferencedFile:
    kind: str
    path: Path
    required: bool = True


@dataclass(frozen=True)
class CleanerConfigInspection:
    config: CleanerConfig
    input_files: tuple[Path, ...]
    referenced_files: tuple[CleanerReferencedFile, ...]
