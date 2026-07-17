from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from nlpo_toolkit.immutable_collections import freeze_count_mapping, freeze_mapping

from .analysis_cache.stats import AnalysisCacheStatsSnapshot
from .token_artifact.schema import TokenArtifactMetadata


@dataclass(frozen=True)
class GroupAnalysisResult:
    files: tuple[Path, ...]
    counter: Mapping[str, int]
    ref_tag_counts: Mapping[str, int]
    token_artifact: TokenArtifactMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "counter", freeze_count_mapping(self.counter))
        object.__setattr__(self, "ref_tag_counts", freeze_count_mapping(self.ref_tag_counts))


@dataclass(frozen=True)
class AnalysisResults:
    groups: Mapping[str, GroupAnalysisResult]
    cache_stats: AnalysisCacheStatsSnapshot

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", freeze_mapping(self.groups))

    @classmethod
    def from_groups(
        cls,
        groups: Iterable[tuple[str, GroupAnalysisResult]],
        *,
        cache_stats: AnalysisCacheStatsSnapshot,
    ) -> AnalysisResults:
        indexed: dict[str, GroupAnalysisResult] = {}
        for label, result in groups:
            if label in indexed:
                raise ValueError(f"Duplicate group analysis result label: {label}")
            indexed[label] = result
        return cls(groups=indexed, cache_stats=cache_stats)
