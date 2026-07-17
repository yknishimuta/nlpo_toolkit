from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AnalysisRecordCacheStatus = Literal["hit", "miss", "disabled"]


@dataclass(frozen=True)
class AnalysisCacheGroupResult:
    group: str
    status: AnalysisRecordCacheStatus
    cache_key: str
    record_count: int


@dataclass(frozen=True)
class AnalysisCacheStatsSnapshot:
    enabled: bool
    directory: str
    hits: int
    misses: int
    objects_written: int
    records_read: int
    records_written: int
    groups: tuple[AnalysisCacheGroupResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))


@dataclass
class AnalysisCacheStatsCollector:
    enabled: bool
    directory: str
    hits: int = 0
    misses: int = 0
    objects_written: int = 0
    records_read: int = 0
    records_written: int = 0
    groups: list[AnalysisCacheGroupResult] = field(default_factory=list)

    def record_group(
        self,
        *,
        group: str,
        status: AnalysisRecordCacheStatus,
        cache_key: str,
        record_count: int,
    ) -> None:
        if status == "hit":
            self.hits += 1
            self.records_read += record_count
        elif status == "miss":
            self.misses += 1
            self.objects_written += 1
            self.records_written += record_count
        self.groups.append(AnalysisCacheGroupResult(group, status, cache_key, record_count))

    def snapshot(self) -> AnalysisCacheStatsSnapshot:
        return AnalysisCacheStatsSnapshot(
            self.enabled, self.directory, self.hits, self.misses,
            self.objects_written, self.records_read, self.records_written,
            tuple(self.groups),
        )
