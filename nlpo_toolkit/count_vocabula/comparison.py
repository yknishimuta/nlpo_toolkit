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
    if count_a < 0 or count_b < 0:
        raise ValueError("counts must be >= 0")
    if tokens_a <= 0 or tokens_b <= 0:
        raise ValueError("tokens must be > 0")
    if not math.isfinite(float(zero_correction)) or zero_correction <= 0:
        raise ValueError("zero_correction must be a positive finite number")

    adjusted_a = count_a if count_a > 0 else float(zero_correction)
    adjusted_b = count_b if count_b > 0 else float(zero_correction)
    relative_a = adjusted_a / tokens_a
    relative_b = adjusted_b / tokens_b
    return math.log2(relative_a / relative_b)


def calculate_log_likelihood(
    *,
    count_a: int,
    count_b: int,
    tokens_a: int,
    tokens_b: int,
) -> float:
    if count_a < 0 or count_b < 0:
        raise ValueError("counts must be >= 0")
    if tokens_a <= 0 or tokens_b <= 0:
        raise ValueError("tokens must be > 0")
    if count_a > tokens_a or count_b > tokens_b:
        raise ValueError("counts must not exceed tokens")

    observed = (
        float(count_a),
        float(tokens_a - count_a),
        float(count_b),
        float(tokens_b - count_b),
    )
    row_totals = (float(tokens_a), float(tokens_b))
    col_totals = (float(count_a + count_b), float(tokens_a + tokens_b - count_a - count_b))
    grand_total = float(tokens_a + tokens_b)

    total = 0.0
    for row_index, row_total in enumerate(row_totals):
        for col_index, col_total in enumerate(col_totals):
            obs = observed[row_index * 2 + col_index]
            if obs <= 0:
                continue
            expected = row_total * col_total / grand_total
            if expected > 0:
                total += obs * math.log(obs / expected)

    g2 = 2.0 * total
    return 0.0 if g2 < 0 and abs(g2) < EPSILON else max(0.0, g2)


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

    all_items = set(counter_a) | set(counter_b)
    rows_before_filter = len(all_items)
    rows: list[ComparisonRow] = []

    for item in all_items:
        count_a = int(counter_a.get(item, 0))
        count_b = int(counter_b.get(item, 0))
        total_count = count_a + count_b
        if total_count < spec.min_total_count:
            continue

        rate_a = count_a / tokens_a * spec.scale
        rate_b = count_b / tokens_b * spec.scale
        log_ratio = calculate_log_ratio(
            count_a=count_a,
            count_b=count_b,
            tokens_a=tokens_a,
            tokens_b=tokens_b,
            zero_correction=spec.zero_correction,
        )
        log_likelihood = calculate_log_likelihood(
            count_a=count_a,
            count_b=count_b,
            tokens_a=tokens_a,
            tokens_b=tokens_b,
        )
        if log_ratio > EPSILON:
            direction = spec.group_a
        elif log_ratio < -EPSILON:
            direction = spec.group_b
        else:
            direction = "equal"

        rows.append(
            ComparisonRow(
                item=str(item),
                group_a_count=count_a,
                group_b_count=count_b,
                group_a_tokens=tokens_a,
                group_b_tokens=tokens_b,
                scale=spec.scale,
                group_a_rate=rate_a,
                group_b_rate=rate_b,
                rate_difference=rate_a - rate_b,
                log_ratio=log_ratio,
                log_likelihood=log_likelihood,
                direction=direction,
                total_count=total_count,
            )
        )

    rows = _sort_rows(rows, spec)
    return ComparisonResult(
        spec=spec,
        analysis_unit=analysis_unit,
        group_a_tokens=tokens_a,
        group_b_tokens=tokens_b,
        vocabulary_union_size=len(all_items),
        rows_before_filter=rows_before_filter,
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
