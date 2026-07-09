from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ON_MISMATCH_VALUES = {"error", "warn"}
REPORT_VALUES = {"mismatches", "all"}
_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z]+")


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    whole: str
    parts: tuple[str, ...]
    on_mismatch: str
    report: str


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


def parse_partition_specs(config: dict) -> list[PartitionSpec]:
    if hasattr(config, "partition_validations"):
        return list(getattr(config, "partition_validations"))

    validations = config.get("validations")
    if validations is None:
        return []
    if not isinstance(validations, dict):
        raise ValueError("'validations' must be a mapping.")

    raw_partitions = validations.get("partitions")
    if raw_partitions is None:
        return []
    if not isinstance(raw_partitions, list):
        raise ValueError("'validations.partitions' must be a list.")

    groups = config.get("groups") or {}
    if not isinstance(groups, dict):
        raise ValueError("Config 'groups' must be a mapping.")
    group_names = set(groups)

    specs: list[PartitionSpec] = []
    seen_names: set[str] = set()

    for index, raw in enumerate(raw_partitions):
        label = f"validations.partitions[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be a mapping.")

        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label}.name must be a non-empty string.")
        name = name.strip()
        if name in seen_names:
            raise ValueError(f"Duplicate partition name: {name}")
        seen_names.add(name)

        whole = raw.get("whole")
        if not isinstance(whole, str) or not whole.strip():
            raise ValueError(f"{label}.whole must be a non-empty string.")
        whole = whole.strip()

        parts_raw = raw.get("parts")
        if not isinstance(parts_raw, list) or not all(
            isinstance(part, str) and part.strip() for part in parts_raw
        ):
            raise ValueError(f"{label}.parts must be list[str].")
        parts = tuple(part.strip() for part in parts_raw)
        if len(parts) < 2:
            raise ValueError(f"{label}.parts must contain at least 2 groups.")
        if len(set(parts)) != len(parts):
            raise ValueError(f"{label}.parts must not contain duplicate group names.")
        if whole in parts:
            raise ValueError(f"{label}.whole must not be included in parts.")

        on_mismatch = raw.get("on_mismatch", "warn")
        if on_mismatch not in ON_MISMATCH_VALUES:
            raise ValueError(f"{label}.on_mismatch must be 'warn' or 'error'.")

        report = raw.get("report", "mismatches")
        if report not in REPORT_VALUES:
            raise ValueError(f"{label}.report must be 'mismatches' or 'all'.")

        missing = [group for group in (whole, *parts) if group not in group_names]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Partition {name} references unknown group(s): {joined}")

        specs.append(
            PartitionSpec(
                name=name,
                whole=whole,
                parts=parts,
                on_mismatch=str(on_mismatch),
                report=str(report),
            )
        )

    return specs


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


def write_partition_validation_csv(
    path: Path,
    result: PartitionResult,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["item", "whole_count"]
        + [f"{part}_count" for part in result.parts]
        + ["parts_sum", "delta", "status"]
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.mismatches:
            out = {
                "item": row.item,
                "whole_count": row.whole_count,
                "parts_sum": row.parts_sum,
                "delta": row.delta,
                "status": row.status,
            }
            for part in result.parts:
                out[f"{part}_count"] = row.part_counts.get(part, 0)
            writer.writerow(out)
    return path


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


def write_partition_validation_json(
    path: Path,
    summaries: Sequence[dict[str, Any]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"partitions": list(summaries)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
