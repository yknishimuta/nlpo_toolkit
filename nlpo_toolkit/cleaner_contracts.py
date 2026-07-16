from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol


CleanerKind = Literal["corpus_corporum", "scholastic_text"]
CLEANER_KINDS: frozenset[str] = frozenset({"corpus_corporum", "scholastic_text"})


class CleanerApplicationError(Exception):
    """Base class for all cleaner application failures."""


class CleanerConfigError(CleanerApplicationError, ValueError):
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


@dataclass(frozen=True)
class CleanerExecutionRequest:
    inspection: CleanerConfigInspection


@dataclass(frozen=True)
class CleanedFileResult:
    input_path: Path
    output_path: Path
    doc_id: str
    reference_event_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", self.input_path.resolve())
        object.__setattr__(self, "output_path", self.output_path.resolve())


@dataclass(frozen=True)
class CleanerExecutionResult:
    config_path: Path
    kind: CleanerKind
    configured_output_path: Path
    files: tuple[CleanedFileResult, ...]
    ref_tsv_path: Path | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", self.config_path.resolve())
        object.__setattr__(
            self, "configured_output_path", self.configured_output_path.resolve()
        )
        object.__setattr__(self, "files", tuple(self.files))
        if self.ref_tsv_path is not None:
            object.__setattr__(self, "ref_tsv_path", self.ref_tsv_path.resolve())

    @property
    def output_files(self) -> tuple[Path, ...]:
        return tuple(file.output_path for file in self.files)

    @property
    def reference_event_count(self) -> int:
        return sum(file.reference_event_count for file in self.files)


class CleanerApplicationService(Protocol):
    def __call__(
        self, request: CleanerExecutionRequest
    ) -> CleanerExecutionResult: ...
