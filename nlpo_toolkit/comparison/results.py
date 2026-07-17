from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from nlpo_toolkit.immutable_collections import freeze_mapping
from .config import ComparisonSpec
from .models import FrequencyTable


@dataclass(frozen=True)
class PairwiseComparisonRow:
    item: str
    count_a: float
    count_b: float
    total_count: float
    rate_a: float
    rate_b: float
    rate_difference: float
    ratio: float
    log_ratio: float
    log_likelihood: float
    direction: str


@dataclass(frozen=True)
class PairwiseComparisonResult:
    table_a: FrequencyTable
    table_b: FrequencyTable
    scale: float
    vocabulary_union_size: int
    rows_before_filter: int
    rows: tuple[PairwiseComparisonRow, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))


@dataclass(frozen=True)
class MultiComparisonRow:
    item: str
    counts: Mapping[str, float]
    rates: Mapping[str, float]
    total_count: float
    max_label: str
    max_rate: float
    min_label: str
    min_rate: float
    range_relative: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", freeze_mapping(self.counts))
        object.__setattr__(self, "rates", freeze_mapping(self.rates))


@dataclass(frozen=True)
class MultiComparisonResult:
    tables: tuple[FrequencyTable, ...]
    scale: float
    vocabulary_union_size: int
    rows_before_filter: int
    rows: tuple[MultiComparisonRow, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tables", tuple(self.tables))
        object.__setattr__(self, "rows", tuple(self.rows))


@dataclass(frozen=True)
class ConfiguredComparisonResult:
    spec: ComparisonSpec
    analysis_unit: str
    comparison: PairwiseComparisonResult

    @property
    def rows(self) -> tuple[PairwiseComparisonRow, ...]:
        return self.comparison.rows

    @property
    def group_a_tokens(self) -> float:
        return self.comparison.table_a.total

    @property
    def group_b_tokens(self) -> float:
        return self.comparison.table_b.total

    @property
    def vocabulary_union_size(self) -> int:
        return self.comparison.vocabulary_union_size

    @property
    def rows_before_filter(self) -> int:
        return self.comparison.rows_before_filter

    @property
    def rows_after_filter(self) -> int:
        return len(self.comparison.rows)


@dataclass(frozen=True)
class CsvPairComparisonResult:
    comparison: PairwiseComparisonResult

    @property
    def rows(self) -> tuple[PairwiseComparisonRow, ...]:
        return self.comparison.rows


@dataclass(frozen=True)
class CsvMultiComparisonResult:
    comparison: MultiComparisonResult

    @property
    def rows(self) -> tuple[MultiComparisonRow, ...]:
        return self.comparison.rows
