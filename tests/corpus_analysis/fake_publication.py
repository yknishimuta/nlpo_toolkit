from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from nlpo_toolkit.corpus_analysis.analysis_records import TokenRecord
from nlpo_toolkit.corpus_analysis.publication_models import (
    ComparisonArtifactPublication,
    GroupArtifactPublication,
    PartitionArtifactPublication,
    RecordArtifactPublicationRequest,
    RunReportPublication,
)
from nlpo_toolkit.corpus_analysis.publication_ports import (
    CountPublicationDependencies,
)
from nlpo_toolkit.corpus_analysis.token_artifact.schema import TokenArtifactMetadata


@dataclass
class RecordingGroupArtifactPublisher:
    calls: list[GroupArtifactPublication] = field(default_factory=list)

    def __call__(self, request: GroupArtifactPublication) -> None:
        self.calls.append(request)


@dataclass
class RecordingPartitionArtifactPublisher:
    calls: list[PartitionArtifactPublication] = field(default_factory=list)

    def __call__(self, request: PartitionArtifactPublication) -> None:
        self.calls.append(request)


@dataclass
class RecordingComparisonArtifactPublisher:
    calls: list[ComparisonArtifactPublication] = field(default_factory=list)

    def __call__(self, request: ComparisonArtifactPublication) -> None:
        self.calls.append(request)


@dataclass
class RecordingRunReportPublisher:
    calls: list[RunReportPublication] = field(default_factory=list)

    def __call__(self, request: RunReportPublication) -> None:
        self.calls.append(request)


@dataclass
class InMemoryRecordArtifactSession:
    records: list[TokenRecord] = field(default_factory=list)
    final_metadata: TokenArtifactMetadata | None = None
    exited_normally: bool = False
    exited_with_exception: bool = False

    def write(self, record: TokenRecord) -> None:
        self.records.append(record)

    @property
    def token_artifact_metadata(self) -> TokenArtifactMetadata | None:
        return self.final_metadata


@dataclass
class InMemoryRecordArtifactSessionFactory:
    requests: list[RecordArtifactPublicationRequest] = field(default_factory=list)
    sessions: list[InMemoryRecordArtifactSession] = field(default_factory=list)

    @contextmanager
    def __call__(
        self, request: RecordArtifactPublicationRequest
    ) -> Iterator[InMemoryRecordArtifactSession]:
        self.requests.append(request)
        session = InMemoryRecordArtifactSession()
        self.sessions.append(session)
        try:
            yield session
        except BaseException:
            session.exited_with_exception = True
            raise
        else:
            session.exited_normally = True


def recording_publication_dependencies() -> CountPublicationDependencies:
    return CountPublicationDependencies(
        group_artifacts=RecordingGroupArtifactPublisher(),
        partition_artifacts=RecordingPartitionArtifactPublisher(),
        comparison_artifacts=RecordingComparisonArtifactPublisher(),
        run_report=RecordingRunReportPublisher(),
        record_artifacts=InMemoryRecordArtifactSessionFactory(),
    )

