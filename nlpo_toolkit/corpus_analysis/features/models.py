from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import math
from typing import Literal

from ..analysis_records import NLPAnalysisRecord
from ..corpus import PreparedCorpus
from ..requests import CorpusPreparationRequest
from .errors import FeatureError
from nlpo_toolkit.immutable_collections import freeze_mapping


FeatureField = Literal["lemma", "token"]
FeatureScalar = str | int | float


@dataclass(frozen=True)
class LexicalDiversityOptions:
    window_size: int = 100
    mtld_threshold: float = 0.72
    hdd_sample_size: int = 42

    def __post_init__(self) -> None:
        for name, value in (
            ("window_size", self.window_size),
            ("hdd_sample_size", self.hdd_sample_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise FeatureError(f"{name} must be a positive integer")
        threshold = self.mtld_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0.0 < threshold < 1.0
        ):
            raise FeatureError("mtld_threshold must be a finite number between 0 and 1")
        object.__setattr__(self, "mtld_threshold", float(threshold))


@dataclass(frozen=True)
class FeatureSamplingOptions:
    window_tokens: int | None = None
    step_tokens: int | None = None
    include_partial: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("window_tokens", self.window_tokens),
            ("step_tokens", self.step_tokens),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise FeatureError(f"{name} must be a positive integer")
        if self.window_tokens is None and self.step_tokens is not None:
            raise FeatureError("step_tokens requires window_tokens")
        if not isinstance(self.include_partial, bool):
            raise FeatureError("include_partial must be a bool")

    @property
    def enabled(self) -> bool:
        return self.window_tokens is not None

    @property
    def effective_step_tokens(self) -> int | None:
        if self.window_tokens is None:
            return None
        return self.step_tokens or self.window_tokens


@dataclass(frozen=True)
class FeatureFilterPolicy:
    min_token_length: int = 0
    drop_roman_numerals: bool = False
    roman_exceptions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if isinstance(self.min_token_length, bool) or not isinstance(
            self.min_token_length, int
        ):
            raise TypeError("min_token_length must be an integer")
        if self.min_token_length < 0:
            raise ValueError("min_token_length must be non-negative")
        if not isinstance(self.drop_roman_numerals, bool):
            raise TypeError("drop_roman_numerals must be a bool")
        object.__setattr__(
            self,
            "roman_exceptions",
            frozenset(
                str(item).strip().lower()
                for item in self.roman_exceptions
                if str(item).strip()
            ),
        )


@dataclass(frozen=True)
class FeatureOptions:
    field: FeatureField = "lemma"
    mfw: int = 0
    include_upos: bool = True
    include_basic: bool = True
    filter_policy: FeatureFilterPolicy = FeatureFilterPolicy()
    sampling: FeatureSamplingOptions = FeatureSamplingOptions()
    lexical_diversity: LexicalDiversityOptions | None = None


def validate_feature_options(options: FeatureOptions) -> None:
    if options.mfw < 0:
        raise FeatureError("--mfw must be non-negative")
    if options.field not in {"lemma", "token"}:
        raise FeatureError("--field must be 'lemma' or 'token'")


@dataclass(frozen=True)
class FeatureRequest:
    corpus: CorpusPreparationRequest
    field: FeatureField = "lemma"
    mfw: int = 0
    include_upos: bool = True
    include_basic: bool = True
    sampling: FeatureSamplingOptions = FeatureSamplingOptions()
    lexical_diversity: LexicalDiversityOptions | None = None


@dataclass(frozen=True)
class FeatureSampleMetadata:
    source_file: str
    sample_id: str
    sample_index: int
    start_token: int
    end_token: int
    kind: Literal["full", "partial"]


@dataclass(frozen=True)
class AnalyzedFeatureCorpus:
    source: PreparedCorpus
    raw_records: tuple[NLPAnalysisRecord, ...]
    lexical_records: tuple[NLPAnalysisRecord, ...]
    char_count: int | None = None
    sample: FeatureSampleMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_records", tuple(self.raw_records))
        object.__setattr__(self, "lexical_records", tuple(self.lexical_records))

    @property
    def raw_record_count(self) -> int:
        return len(self.raw_records)

    @property
    def lexical_record_count(self) -> int:
        return len(self.lexical_records)

    @property
    def sentence_count(self) -> int:
        return len(
            {(record.chunk_index, record.sentence_index) for record in self.raw_records}
        )


@dataclass(frozen=True)
class FeatureRow(Mapping[str, FeatureScalar]):
    values: Mapping[str, FeatureScalar]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_mapping(self.values))

    @classmethod
    def from_mapping(cls, values: Mapping[str, FeatureScalar]) -> FeatureRow:
        return cls(values)

    def __getitem__(self, key: str) -> FeatureScalar:
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class FeatureCommandResult:
    rows: tuple[FeatureRow, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
