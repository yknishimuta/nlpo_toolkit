from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from ..analysis_records import NLPAnalysisRecord
from ..corpus import PreparedCorpus
from ..requests import CorpusPreparationRequest
from .errors import FeatureError
from nlpo_toolkit.immutable_collections import freeze_mapping


FeatureField = Literal["lemma", "token"]
FeatureScalar = str | int | float


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


@dataclass(frozen=True)
class AnalyzedFeatureCorpus:
    source: PreparedCorpus
    raw_record_count: int
    sentence_count: int
    records: tuple[NLPAnalysisRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))


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
