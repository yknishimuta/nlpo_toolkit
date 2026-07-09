from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Mapping, Sequence

from nlpo_toolkit.comparison import (
    ComparisonEngineError,
    FrequencyTable,
    PairwiseComparisonOptions,
    ZeroHandling,
    ZeroHandlingMode,
    calculate_log_likelihood as engine_log_likelihood,
    calculate_log_ratio as engine_log_ratio,
    compare_pair,
)


REPORT_VALUES = {"all", "filtered"}
SORT_BY_VALUES = {"log_likelihood", "abs_log_ratio", "total_count", "item"}
EPSILON = 1e-12
_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z]+")


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    group_a: str
    group_b: str
    scale: int = 10_000
    zero_correction: float = 0.5
    min_total_count: int = 1
    report: str = "all"
    sort_by: str = "log_likelihood"
    sort_descending: bool = True


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


def _is_int_not_bool(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def parse_comparison_specs(config: Mapping[str, Any]) -> list[ComparisonSpec]:
    if hasattr(config, "comparisons"):
        return list(getattr(config, "comparisons"))

    raw_comparisons = config.get("comparisons")
    if raw_comparisons is None:
        return []
    if not isinstance(raw_comparisons, list):
        raise ValueError("'comparisons' must be a list.")

    grouping = config.get("grouping") or {}
    if isinstance(grouping, Mapping) and grouping.get("mode", "groups") == "per_file":
        raise ValueError("comparisons cannot be used with grouping.mode=per_file")

    groups = config.get("groups") or {}
    if not isinstance(groups, Mapping):
        raise ValueError("Config 'groups' must be a mapping.")
    group_names = set(groups)

    specs: list[ComparisonSpec] = []
    seen_names: set[str] = set()

    for index, raw in enumerate(raw_comparisons):
        label = f"comparisons[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be a mapping.")

        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label}.name must be a non-empty string.")
        name = name.strip()
        if name in seen_names:
            raise ValueError(f"Duplicate comparison name: {name}")
        seen_names.add(name)

        group_a = raw.get("group_a")
        if not isinstance(group_a, str) or not group_a.strip():
            raise ValueError(f"comparison '{name}': group_a must be a non-empty string.")
        group_a = group_a.strip()

        group_b = raw.get("group_b")
        if not isinstance(group_b, str) or not group_b.strip():
            raise ValueError(f"comparison '{name}': group_b must be a non-empty string.")
        group_b = group_b.strip()

        if group_a == group_b:
            raise ValueError(f"comparison '{name}': group_a and group_b must be different.")
        if group_a not in group_names:
            raise ValueError(f"comparison '{name}': unknown group_a '{group_a}'")
        if group_b not in group_names:
            raise ValueError(f"comparison '{name}': unknown group_b '{group_b}'")

        scale = raw.get("scale", 10_000)
        if not _is_int_not_bool(scale) or scale <= 0:
            raise ValueError(f"comparison '{name}': scale must be a positive integer.")

        zero_correction = raw.get("zero_correction", 0.5)
        if not _is_positive_finite_number(zero_correction):
            raise ValueError(f"comparison '{name}': zero_correction must be a positive finite number.")

        min_total_count = raw.get("min_total_count", 1)
        if not _is_int_not_bool(min_total_count) or min_total_count < 1:
            raise ValueError(f"comparison '{name}': min_total_count must be an integer >= 1.")

        report = raw.get("report", "all")
        if report not in REPORT_VALUES:
            raise ValueError(f"comparison '{name}': report must be 'all' or 'filtered'.")

        sort_by = "log_likelihood"
        sort_descending = True
        sort = raw.get("sort")
        if sort is not None:
            if not isinstance(sort, Mapping):
                raise ValueError(f"comparison '{name}': sort must be a mapping.")
            sort_by = sort.get("by", "log_likelihood")
            if sort_by not in SORT_BY_VALUES:
                raise ValueError(f"comparison '{name}': sort.by must be one of {sorted(SORT_BY_VALUES)}.")
            sort_descending = sort.get("descending", True)
            if not isinstance(sort_descending, bool):
                raise ValueError(f"comparison '{name}': sort.descending must be bool.")

        specs.append(
            ComparisonSpec(
                name=name,
                group_a=group_a,
                group_b=group_b,
                scale=int(scale),
                zero_correction=float(zero_correction),
                min_total_count=int(min_total_count),
                report=str(report),
                sort_by=str(sort_by),
                sort_descending=bool(sort_descending),
            )
        )

    return specs


def calculate_log_ratio(
    *,
    count_a: int,
    count_b: int,
    tokens_a: int,
    tokens_b: int,
    zero_correction: float,
) -> float:
    try:
        return engine_log_ratio(
            count_a=float(count_a),
            count_b=float(count_b),
            total_a=float(tokens_a),
            total_b=float(tokens_b),
            zero_handling=ZeroHandling(ZeroHandlingMode.ZERO_ONLY, zero_correction),
        )
    except ComparisonEngineError as exc:
        raise ValueError(str(exc)) from exc


def calculate_log_likelihood(
    *,
    count_a: int,
    count_b: int,
    tokens_a: int,
    tokens_b: int,
) -> float:
    try:
        return engine_log_likelihood(
            count_a=float(count_a),
            count_b=float(count_b),
            total_a=float(tokens_a),
            total_b=float(tokens_b),
        )
    except ComparisonEngineError as exc:
        raise ValueError(str(exc)) from exc


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
            _primary_value(left, spec.sort_by),
            _primary_value(right, spec.sort_by),
            descending=spec.sort_descending,
        )
        if primary:
            return primary

        for key, descending in (
            ("log_likelihood", True),
            ("abs_log_ratio", True),
            ("total_count", True),
            ("item", False),
        ):
            if key == spec.sort_by:
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


def comparison_csv_name(spec: ComparisonSpec) -> str:
    return f"group_comparison_{sanitize_comparison_name(spec.name)}.csv"


def write_comparison_csv(path: Path, result: ComparisonResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "comparison",
        "analysis_unit",
        "item",
        "group_a",
        "group_b",
        "group_a_count",
        "group_b_count",
        "group_a_tokens",
        "group_b_tokens",
        "scale",
        "group_a_rate",
        "group_b_rate",
        "rate_difference",
        "log_ratio",
        "log_likelihood",
        "direction",
        "total_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.rows:
            writer.writerow(
                {
                    "comparison": result.spec.name,
                    "analysis_unit": result.analysis_unit,
                    "item": row.item,
                    "group_a": result.spec.group_a,
                    "group_b": result.spec.group_b,
                    "group_a_count": row.group_a_count,
                    "group_b_count": row.group_b_count,
                    "group_a_tokens": row.group_a_tokens,
                    "group_b_tokens": row.group_b_tokens,
                    "scale": row.scale,
                    "group_a_rate": f"{row.group_a_rate:.6f}",
                    "group_b_rate": f"{row.group_b_rate:.6f}",
                    "rate_difference": f"{row.rate_difference:.6f}",
                    "log_ratio": f"{row.log_ratio:.6f}",
                    "log_likelihood": f"{row.log_likelihood:.6f}",
                    "direction": row.direction,
                    "total_count": row.total_count,
                }
            )
    return path


def comparison_result_summary(result: ComparisonResult, *, csv_name: str) -> dict[str, Any]:
    spec = result.spec
    return {
        "name": spec.name,
        "group_a": spec.group_a,
        "group_b": spec.group_b,
        "scale": spec.scale,
        "zero_correction": spec.zero_correction,
        "min_total_count": spec.min_total_count,
        "group_a_tokens": result.group_a_tokens,
        "group_b_tokens": result.group_b_tokens,
        "vocabulary_union_size": result.vocabulary_union_size,
        "rows_before_filter": result.rows_before_filter,
        "rows_after_filter": result.rows_after_filter,
        "csv": csv_name,
    }


def comparison_result_meta(result: ComparisonResult, *, csv_name: str) -> dict[str, Any]:
    data = comparison_result_summary(result, csv_name=csv_name)
    data["analysis_unit"] = result.analysis_unit
    data.pop("rows_before_filter", None)
    return data


def write_group_comparisons_json(path: Path, results: Sequence[ComparisonResult]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    analysis_unit = results[0].analysis_unit if results else ""
    data = {
        "analysis_unit": analysis_unit,
        "comparisons": [
            comparison_result_summary(result, csv_name=comparison_csv_name(result.spec))
            for result in results
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
