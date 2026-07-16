from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Annotated, Any, Literal, Mapping, Sequence

from pydantic import Field, StrictBool, StrictInt, model_validator

from nlpo_toolkit.config_model import (
    ConfigModel,
    PositiveFiniteFloat,
    StrippedNonBlankStr,
)

from nlpo_toolkit.comparison import (
    ComparisonEngineError,
    FrequencyTable,
    PairwiseComparisonOptions,
    ZeroHandling,
    ZeroHandlingMode,
    compare_pair,
)


EPSILON = 1e-12
_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z]+")


ComparisonSortBy = Literal[
    "log_likelihood",
    "abs_log_ratio",
    "total_count",
    "item",
]
PositiveStrictInt = Annotated[StrictInt, Field(gt=0)]


class ComparisonSortConfig(ConfigModel):
    by: ComparisonSortBy = "log_likelihood"
    descending: StrictBool = True


class ComparisonSpec(ConfigModel):
    name: StrippedNonBlankStr
    group_a: StrippedNonBlankStr
    group_b: StrippedNonBlankStr
    scale: PositiveStrictInt = 10_000
    zero_correction: PositiveFiniteFloat = 0.5
    min_total_count: PositiveStrictInt = 1
    sort: ComparisonSortConfig = Field(default_factory=ComparisonSortConfig)

    @model_validator(mode="after")
    def validate_distinct_groups(self) -> ComparisonSpec:
        if self.group_a == self.group_b:
            raise ValueError("group_a and group_b must be different")
        return self


@dataclass(frozen=True)
class ComparisonRow:
    item: str
    group_a_count: int
    group_b_count: int
    group_a_tokens: int
    group_b_tokens: int
    scale: int
    group_a_rate: float
    group_b_rate: float
    rate_difference: float
    log_ratio: float
    log_likelihood: float
    direction: str
    total_count: int


@dataclass(frozen=True)
class ComparisonResult:
    spec: ComparisonSpec
    analysis_unit: str
    group_a_tokens: int
    group_b_tokens: int
    vocabulary_union_size: int
    rows_before_filter: int
    rows_after_filter: int
    rows: tuple[ComparisonRow, ...]


def sanitize_comparison_name(name: str) -> str:
    safe = _SAFE_NAME_RE.sub("_", name).strip("_").lower()
    return safe or "comparison"


def _primary_value(row: ComparisonRow, sort_by: str) -> float | int | str:
    if sort_by == "log_likelihood":
        return row.log_likelihood
    if sort_by == "abs_log_ratio":
        return abs(row.log_ratio)
    if sort_by == "total_count":
        return row.total_count
    if sort_by == "item":
        return row.item
    raise ValueError(f"unsupported sort.by: {sort_by}")


def _compare_values(a: Any, b: Any, *, descending: bool) -> int:
    if a < b:
        return 1 if descending else -1
    if a > b:
        return -1 if descending else 1
    return 0


def _sort_rows(rows: list[ComparisonRow], spec: ComparisonSpec) -> list[ComparisonRow]:
    def compare(left: ComparisonRow, right: ComparisonRow) -> int:
        primary = _compare_values(
            _primary_value(left, spec.sort.by),
            _primary_value(right, spec.sort.by),
            descending=spec.sort.descending,
        )
        if primary:
            return primary

        for key, descending in (
            ("log_likelihood", True),
            ("abs_log_ratio", True),
            ("total_count", True),
            ("item", False),
        ):
            if key == spec.sort.by:
                continue
            result = _compare_values(
                _primary_value(left, key),
                _primary_value(right, key),
                descending=descending,
            )
            if result:
                return result
        return 0

    return sorted(rows, key=cmp_to_key(compare))


def _frequency_table_from_counter(
    *,
    label: str,
    counter: Counter[str],
    comparison_name: str,
) -> FrequencyTable:
    counts: dict[str, int] = {}
    for item, value in counter.items():
        if not isinstance(item, str):
            raise ValueError(f"comparison '{comparison_name}': counter keys must be strings")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"comparison '{comparison_name}': counter values must be integers")
        counts[item] = value
    try:
        return FrequencyTable.from_counts(label, counts)
    except ComparisonEngineError as exc:
        raise ValueError(str(exc)) from exc


def _int_count(value: float) -> int:
    if not float(value).is_integer():
        raise ValueError("group comparison counts must be integers")
    return int(value)


def compare_counters(
    *,
    counter_a: Counter[str],
    counter_b: Counter[str],
    spec: ComparisonSpec,
    analysis_unit: str,
) -> ComparisonResult:
    tokens_a = sum(int(v) for v in counter_a.values())
    tokens_b = sum(int(v) for v in counter_b.values())
    if tokens_a == 0:
        raise ValueError(f"comparison '{spec.name}': group '{spec.group_a}' has zero target tokens")
    if tokens_b == 0:
        raise ValueError(f"comparison '{spec.name}': group '{spec.group_b}' has zero target tokens")

    table_a = _frequency_table_from_counter(
        label=spec.group_a,
        counter=counter_a,
        comparison_name=spec.name,
    )
    table_b = _frequency_table_from_counter(
        label=spec.group_b,
        counter=counter_b,
        comparison_name=spec.name,
    )
    try:
        engine_result = compare_pair(
            table_a,
            table_b,
            options=PairwiseComparisonOptions(
                scale=spec.scale,
                min_total_count=spec.min_total_count,
                zero_handling=ZeroHandling(ZeroHandlingMode.ZERO_ONLY, spec.zero_correction),
            ),
        )
    except ComparisonEngineError as exc:
        raise ValueError(str(exc)) from exc

    rows: list[ComparisonRow] = []
    for row in engine_result.rows:
        count_a = _int_count(row.count_a)
        count_b = _int_count(row.count_b)
        rows.append(
            ComparisonRow(
                item=row.item,
                group_a_count=count_a,
                group_b_count=count_b,
                group_a_tokens=tokens_a,
                group_b_tokens=tokens_b,
                scale=spec.scale,
                group_a_rate=row.rate_a,
                group_b_rate=row.rate_b,
                rate_difference=row.rate_difference,
                log_ratio=row.log_ratio,
                log_likelihood=row.log_likelihood,
                direction=row.direction,
                total_count=_int_count(row.total_count),
            )
        )

    rows = _sort_rows(rows, spec)
    return ComparisonResult(
        spec=spec,
        analysis_unit=analysis_unit,
        group_a_tokens=tokens_a,
        group_b_tokens=tokens_b,
        vocabulary_union_size=engine_result.vocabulary_union_size,
        rows_before_filter=engine_result.rows_before_filter,
        rows_after_filter=len(rows),
        rows=tuple(rows),
    )


def run_comparisons(
    *,
    specs: Sequence[ComparisonSpec],
    counters: Mapping[str, Counter[str]],
    analysis_unit: str,
) -> list[ComparisonResult]:
    return [
        compare_counters(
            counter_a=counters[spec.group_a],
            counter_b=counters[spec.group_b],
            spec=spec,
            analysis_unit=analysis_unit,
        )
        for spec in specs
    ]
