"""Serialization for configured group comparisons."""

from typing import Any

from .configured import ComparisonResult


def comparison_result_summary(result: ComparisonResult, *, csv_name: str) -> dict[str, Any]:
    spec = result.spec
    return {"name": spec.name, "group_a": spec.group_a, "group_b": spec.group_b, "scale": spec.scale, "zero_correction": spec.zero_correction, "min_total_count": spec.min_total_count, "group_a_tokens": result.group_a_tokens, "group_b_tokens": result.group_b_tokens, "vocabulary_union_size": result.vocabulary_union_size, "rows_before_filter": result.rows_before_filter, "rows_after_filter": result.rows_after_filter, "csv": csv_name}


def comparison_result_meta(result: ComparisonResult, *, csv_name: str) -> dict[str, Any]:
    data = comparison_result_summary(result, csv_name=csv_name)
    data["analysis_unit"] = result.analysis_unit
    data.pop("rows_before_filter", None)
    return data

