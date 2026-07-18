from __future__ import annotations

from dataclasses import dataclass, field

from .analysis_cache_results import (
    AnalysisCacheGroupResult,
    AnalysisCacheStatsSnapshot,
    AnalysisRecordCacheStatus,
)


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
            enabled=self.enabled,
            directory=self.directory,
            hits=self.hits,
            misses=self.misses,
            objects_written=self.objects_written,
            records_read=self.records_read,
            records_written=self.records_written,
            groups=tuple(self.groups),
        )
