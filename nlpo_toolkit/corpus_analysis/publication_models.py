"""Typed values passed from Count application services to publication ports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from nlpo_toolkit.comparison.config import ComparisonSpec
from nlpo_toolkit.comparison.engine import FrequencyTable
from nlpo_toolkit.comparison.results import ConfiguredComparisonResult
from nlpo_toolkit.immutable_collections import freeze_count_mapping

from .artifacts.models import ArtifactPlan, PlannedArtifact
from .partition_models import PartitionSpec
from .partition_validation import PartitionResult
from .postprocessing.dictionary import DictionaryClassification
from .reporting.models import RunMetadata
from .token_artifact.schema import TokenArtifactDescriptor


ConfiguredResult = ConfiguredComparisonResult[ComparisonSpec, FrequencyTable]


@dataclass(frozen=True)
class GroupArtifactPublication:
    artifact_plan: ArtifactPlan
    group: str
    counter: Mapping[str, int]
    dictionary: DictionaryClassification | None
    reference_tag_counts: Mapping[str, int]
    csv_header: tuple[str, str]
    reference_tags_enabled: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "counter", freeze_count_mapping(self.counter))
        object.__setattr__(self, "reference_tag_counts", freeze_count_mapping(self.reference_tag_counts))


@dataclass(frozen=True)
class PartitionArtifactPublication:
    artifact_plan: ArtifactPlan
    specs: tuple[PartitionSpec, ...]
    results: tuple[PartitionResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", tuple(self.specs))
        object.__setattr__(self, "results", tuple(self.results))


@dataclass(frozen=True)
class ComparisonArtifactPublication:
    artifact_plan: ArtifactPlan
    results: tuple[ConfiguredResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))


@dataclass(frozen=True)
class RunReportPublication:
    artifact_plan: ArtifactPlan
    summary: str
    metadata: RunMetadata


@dataclass(frozen=True)
class RecordArtifactPublicationRequest:
    token_artifact: PlannedArtifact | None
    token_artifact_metadata: PlannedArtifact | None
    diagnostic_trace: PlannedArtifact | None
    descriptor: TokenArtifactDescriptor
    trace_max_rows: int
    trace_only_keys: tuple[str, ...]
    trace_write_truncation_marker: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_only_keys", tuple(self.trace_only_keys))
