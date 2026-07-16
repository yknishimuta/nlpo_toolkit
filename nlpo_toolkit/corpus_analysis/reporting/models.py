from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

from nlpo_toolkit.nlp.contracts import NLPBackendInfo

from ..artifacts.models import ArtifactKind
from ..config.models import AnalysisUnit, GroupingMode, NormalizationConfig


@dataclass(frozen=True)
class GroupingReport:
    mode: GroupingMode
    auto_group_name: str | None = None


@dataclass(frozen=True)
class RuntimeEnvironmentReport:
    python_version: str
    platform: str
    executable: Path
    project_root: Path
    git_commit: str | None
    git_status: str | None


@dataclass(frozen=True)
class TraceArtifactReport:
    group: str
    path: Path


@dataclass(frozen=True)
class TokenArtifactReport:
    group: str
    path: Path
    metadata_path: Path
    schema_version: int
    row_count: int
    included_row_count: int
    complete: bool
    sha256: str


@dataclass(frozen=True)
class GeneratedArtifactReport:
    kind: ArtifactKind
    path: Path
    group: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class PartitionReport:
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
    on_mismatch: str


@dataclass(frozen=True)
class ComparisonReport:
    name: str
    group_a: str
    group_b: str
    scale: int
    zero_correction: float
    min_total_count: int
    analysis_unit: str
    group_a_tokens: int
    group_b_tokens: int
    vocabulary_union_size: int
    rows_after_filter: int
    csv_name: str


@dataclass(frozen=True)
class AnalysisCacheGroupReport:
    group: str
    status: str
    cache_key: str
    record_count: int


@dataclass(frozen=True)
class AnalysisCacheReport:
    enabled: bool
    directory: str
    hits: int
    misses: int
    objects_written: int
    records_read: int
    records_written: int
    groups: tuple[AnalysisCacheGroupReport, ...]


@dataclass(frozen=True)
class RunMetadata:
    generated_at: datetime
    groups_files: Mapping[str, tuple[Path, ...]]
    analysis_unit: AnalysisUnit
    nlp: NLPBackendInfo
    grouping: GroupingReport
    environment: RuntimeEnvironmentReport
    normalization: NormalizationConfig
    normalization_hash_sha256: str
    partition_validations: tuple[PartitionReport, ...]
    group_comparisons: tuple[ComparisonReport, ...]
    traces: tuple[TraceArtifactReport, ...]
    token_artifacts: tuple[TokenArtifactReport, ...]
    analysis_cache: AnalysisCacheReport
    generated_artifacts: tuple[GeneratedArtifactReport, ...]

    @property
    def generated_outputs(self) -> tuple[Path, ...]:
        return tuple(artifact.path for artifact in self.generated_artifacts)
