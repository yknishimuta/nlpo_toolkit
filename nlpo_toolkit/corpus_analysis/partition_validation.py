from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from nlpo_toolkit.immutable_collections import freeze_count_mapping


from .partition_models import PartitionSpec
_SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z]+")


@dataclass(frozen=True)
class PartitionMismatch:
    item: str
    whole_count: int
    part_counts: Mapping[str, int]
    parts_sum: int
    delta: int
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "part_counts", freeze_count_mapping(self.part_counts))


@dataclass(frozen=True)
class PartitionResult:
    name: str
    whole: str
    parts: tuple[str, ...]
    exact_match: bool
    whole_target_tokens: int
    parts_target_tokens: int
    token_delta: int
    whole_types: int
    parts_union_types: int
    mismatched_items: int
    matched_items: int
    mismatches: tuple[PartitionMismatch, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parts", tuple(self.parts))
        object.__setattr__(self, "mismatches", tuple(self.mismatches))


def sanitize_partition_name(name: str) -> str:
    safe = _SAFE_NAME_RE.sub("_", name).strip("_").lower()
    return safe or "partition"


def validate_partition(
    spec: PartitionSpec,
    counters: Mapping[str, Mapping[str, int]],
) -> PartitionResult:
    whole_counter = Counter(counters[spec.whole])
    parts_total: Counter[str] = Counter()
    for name in spec.parts:
        parts_total.update(counters[name])
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
        parts=tuple(spec.parts),
        exact_match=whole_counter == parts_total,
        whole_target_tokens=whole_target_tokens,
        parts_target_tokens=parts_target_tokens,
        token_delta=whole_target_tokens - parts_target_tokens,
        whole_types=len(whole_counter),
        parts_union_types=len(parts_total),
        mismatched_items=mismatched_items,
        matched_items=matched_items,
        mismatches=tuple(rows),
    )


def validate_partitions(
    specs: Sequence[PartitionSpec],
    counters: Mapping[str, Mapping[str, int]],
) -> tuple[PartitionResult, ...]:
    return tuple(validate_partition(spec, counters) for spec in specs)
