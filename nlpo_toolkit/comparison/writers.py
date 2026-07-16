"""Serialization for configured group comparisons."""

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .configured import ComparisonResult


def write_group_comparison_csv(path: Path, result: ComparisonResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["comparison", "analysis_unit", "item", "group_a", "group_b", "group_a_count", "group_b_count", "group_a_tokens", "group_b_tokens", "scale", "group_a_rate", "group_b_rate", "rate_difference", "log_ratio", "log_likelihood", "direction", "total_count"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.rows:
            writer.writerow({"comparison": result.spec.name, "analysis_unit": result.analysis_unit, "item": row.item, "group_a": result.spec.group_a, "group_b": result.spec.group_b, "group_a_count": row.group_a_count, "group_b_count": row.group_b_count, "group_a_tokens": row.group_a_tokens, "group_b_tokens": row.group_b_tokens, "scale": row.scale, "group_a_rate": f"{row.group_a_rate:.6f}", "group_b_rate": f"{row.group_b_rate:.6f}", "rate_difference": f"{row.rate_difference:.6f}", "log_ratio": f"{row.log_ratio:.6f}", "log_likelihood": f"{row.log_likelihood:.6f}", "direction": row.direction, "total_count": row.total_count})
    return path


def comparison_result_summary(result: ComparisonResult, *, csv_name: str) -> dict[str, Any]:
    spec = result.spec
    return {"name": spec.name, "group_a": spec.group_a, "group_b": spec.group_b, "scale": spec.scale, "zero_correction": spec.zero_correction, "min_total_count": spec.min_total_count, "group_a_tokens": result.group_a_tokens, "group_b_tokens": result.group_b_tokens, "vocabulary_union_size": result.vocabulary_union_size, "rows_before_filter": result.rows_before_filter, "rows_after_filter": result.rows_after_filter, "csv": csv_name}


def comparison_result_meta(result: ComparisonResult, *, csv_name: str) -> dict[str, Any]:
    data = comparison_result_summary(result, csv_name=csv_name)
    data["analysis_unit"] = result.analysis_unit
    data.pop("rows_before_filter", None)
    return data


def write_group_comparisons_json(path: Path, results: Sequence[ComparisonResult], *,
                                 csv_names: Mapping[str, str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"analysis_unit": results[0].analysis_unit if results else "", "comparisons": [comparison_result_summary(result, csv_name=csv_names[result.spec.name]) for result in results]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
