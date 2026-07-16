from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


from .partition_models import PartitionSpec
_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z]+")


@dataclass(frozen=True)
class PartitionMismatch:
    item: str
    whole_count: int
    part_counts: dict[str, int]
    parts_sum: int
    delta: int
    status: str


@dataclass
class PartitionResult:
    name: str
    whole: str
    parts: list[str]
    exact_match: bool
    whole_target_tokens: int
    parts_target_tokens: int
    token_delta: int
    whole_types: int
    parts_union_types: int
    mismatched_items: int
    matched_items: int
    mismatches: list[PartitionMismatch]


def sanitize_partition_name(name: str) -> str:
    safe = _SAFE_NAME_RE.sub("_", name).strip("_").lower()
    return safe or "partition"


def validate_partition(
    spec: PartitionSpec,
    counters: Mapping[str, Counter],
) -> PartitionResult:
    whole_counter = counters[spec.whole]
    parts_total = sum((counters[name] for name in spec.parts), Counter())
    all_items = set(whole_counter) | set(parts_total)

    rows: list[PartitionMismatch] = []
    matched_items = 0
    mismatched_items = 0

    for item in all_items:
        whole_count = int(whole_counter.get(item, 0))
        part_counts = {name: int(counters[name].get(item, 0)) for name in spec.parts}
        parts_sum = int(parts_total.get(item, 0))
        delta = whole_count - parts_sum
        if delta == 0:
            status = "match"
            matched_items += 1
        elif delta > 0:
            status = "missing_from_parts"
            mismatched_items += 1
        else:
            status = "excess_in_parts"
            mismatched_items += 1

        if spec.report == "all" or delta != 0:
            rows.append(
                PartitionMismatch(
                    item=str(item),
                    whole_count=whole_count,
                    part_counts=part_counts,
                    parts_sum=parts_sum,
                    delta=delta,
                    status=status,
                )
            )

    rows.sort(key=lambda row: (-abs(row.delta), row.item))

    whole_target_tokens = sum(int(v) for v in whole_counter.values())
    parts_target_tokens = sum(int(v) for v in parts_total.values())

    return PartitionResult(
        name=spec.name,
        whole=spec.whole,
        parts=list(spec.parts),
        exact_match=whole_counter == parts_total,
        whole_target_tokens=whole_target_tokens,
        parts_target_tokens=parts_target_tokens,
        token_delta=whole_target_tokens - parts_target_tokens,
        whole_types=len(whole_counter),
        parts_union_types=len(parts_total),
        mismatched_items=mismatched_items,
        matched_items=matched_items,
        mismatches=rows,
    )


def validate_partitions(
    specs: Sequence[PartitionSpec],
    counters: Mapping[str, Counter],
) -> list[PartitionResult]:
    return [validate_partition(spec, counters) for spec in specs]


def partition_result_summary(
    spec: PartitionSpec,
    result: PartitionResult,
    *,
    csv_name: str,
) -> dict[str, Any]:
    return {
        "name": result.name,
        "whole": result.whole,
        "parts": result.parts,
        "exact_match": result.exact_match,
        "whole_target_tokens": result.whole_target_tokens,
        "parts_target_tokens": result.parts_target_tokens,
        "token_delta": result.token_delta,
        "whole_types": result.whole_types,
        "parts_union_types": result.parts_union_types,
        "matched_items": result.matched_items,
        "mismatched_items": result.mismatched_items,
        "on_mismatch": spec.on_mismatch,
        "csv": csv_name,
    }


def partition_result_meta(spec: PartitionSpec, result: PartitionResult) -> dict[str, Any]:
    data = partition_result_summary(spec, result, csv_name="")
    data.pop("csv", None)
    data.pop("matched_items", None)
    return data

