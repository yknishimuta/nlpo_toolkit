from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nlpo_toolkit.immutable_collections import freeze_count_mapping

from .analysis_cache_stats import AnalysisCacheStatsCollector
from .analysis_records import (
    AnalysisOptions,
    NLPAnalysisRecord,
    evaluate_analysis_record,
)
from .corpus import PreparedCorpus
from .artifacts.models import ArtifactKind
from .publication_models import RecordArtifactPublicationRequest
from .publication_ports import RecordArtifactSession, RecordArtifactSessionFactory
from .count_context import CountRunContext
from .ports import (
    AnalysisRecordCacheSettings,
    AnalysisRecordProvider,
    AnalysisRecordRequest,
    AnalysisRecordSource,
)
from .token_artifact.schema import (
    TokenArtifactDescriptor, TokenArtifactFilterDescriptor,
    TokenArtifactMetadata, TokenArtifactNLPDescriptor,
)

__all__ = [
    "RecordAnalysisResult",
    "RecordConsumptionResult",
    "build_analysis_options",
    "consume_analysis_records",
    "execute_record_analysis",
]


@dataclass(frozen=True)
class RecordConsumptionResult:
    counter: Mapping[str, int]
    record_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "counter", freeze_count_mapping(self.counter))


@dataclass(frozen=True)
class RecordAnalysisResult:
    counter: Mapping[str, int]
    record_count: int
    token_artifact: TokenArtifactMetadata | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "counter", freeze_count_mapping(self.counter))


def _analysis_cache_dir(context: CountRunContext) -> Path:
    definition = context.session.corpus.plan.definition
    directory = Path(definition.config.analysis_cache.directory)
    if not directory.is_absolute():
        directory = definition.project_root / directory
    return directory.resolve()


def analysis_record_cache_settings(
    context: CountRunContext,
) -> AnalysisRecordCacheSettings:
    config = context.session.corpus.plan.definition.config.analysis_cache
    return AnalysisRecordCacheSettings(
        enabled=config.enabled,
        directory=_analysis_cache_dir(context),
        lock_timeout_sec=config.lock_timeout_sec,
    )


def build_analysis_options(
    *,
    context: CountRunContext,
    corpus: PreparedCorpus,
) -> AnalysisOptions:
    definition = context.session.corpus.plan.definition
    filters = definition.config.filters
    return AnalysisOptions(
        group=corpus.label,
        source_files=tuple(corpus.files),
        use_lemma=definition.analysis_mode.use_lemma,
        upos_targets=frozenset(filters.upos_targets),
        min_token_length=filters.min_token_length,
        drop_roman_numerals=filters.drop_roman_numerals,
        roman_exceptions=context.session.roman_exceptions,
    )


def consume_analysis_records(
    *,
    records: Iterable[NLPAnalysisRecord],
    options: AnalysisOptions,
    record_sink: RecordArtifactSession,
) -> RecordConsumptionResult:
    counter: Counter[str] = Counter()
    record_count = 0
    for raw_record in records:
        record_count += 1
        record = evaluate_analysis_record(raw_record, options=options)
        record_sink.write(record)
        if record.included and record.analysis_key:
            counter[record.analysis_key] += 1
    return RecordConsumptionResult(counter=counter, record_count=record_count)


def _token_artifact_descriptor(
    *, context: CountRunContext, corpus: PreparedCorpus
) -> TokenArtifactDescriptor:
    definition = context.session.corpus.plan.definition
    return TokenArtifactDescriptor(
        group=corpus.label,
        source_files=tuple(str(file) for file in corpus.files),
        analysis_unit=definition.analysis_mode.unit,
        upos_targets=tuple(sorted(definition.config.filters.upos_targets)),
        nlp=TokenArtifactNLPDescriptor(
            backend=context.session.backend.info.name,
            language=context.session.backend.info.language,
            model=context.session.backend.info.model,
            package=context.session.backend.info.package,
            device=context.session.backend.info.device,
        ),
        filters=TokenArtifactFilterDescriptor(
            upos_targets=tuple(sorted(definition.config.filters.upos_targets)),
            min_token_length=definition.config.filters.min_token_length,
            drop_roman_numerals=definition.config.filters.drop_roman_numerals,
            roman_exceptions=(),
        ),
    )


def _update_cache_stats(
    *,
    stats: AnalysisCacheStatsCollector,
    corpus: PreparedCorpus,
    source: AnalysisRecordSource,
    record_count: int,
) -> None:
    stats.record_group(
        group=corpus.label,
        status=source.cache_status,
        cache_key=source.cache_key,
        record_count=record_count,
    )


def execute_record_analysis(
    *,
    context: CountRunContext,
    corpus: PreparedCorpus,
    text: str,
    publication: RecordArtifactSessionFactory,
    analysis_records: AnalysisRecordProvider,
    cache_stats: AnalysisCacheStatsCollector,
) -> RecordAnalysisResult:
    artifacts = context.artifact_plan
    trace_config = context.session.corpus.plan.definition.config.trace
    request = RecordArtifactPublicationRequest(
        token_artifact=artifacts.optional(ArtifactKind.TOKEN_ARTIFACT, group=corpus.label),
        token_artifact_metadata=artifacts.optional(
            ArtifactKind.TOKEN_ARTIFACT_METADATA, group=corpus.label
        ),
        diagnostic_trace=artifacts.optional(
            ArtifactKind.DIAGNOSTIC_TRACE, group=corpus.label
        ),
        descriptor=_token_artifact_descriptor(context=context, corpus=corpus),
        trace_max_rows=int(trace_config.max_rows or 0),
        trace_only_keys=tuple(trace_config.only_keys),
        trace_write_truncation_marker=trace_config.write_truncation_marker,
    )
    with publication(request) as record_session:
        record_request = AnalysisRecordRequest(
            text=text,
            backend=context.session.backend,
            extraction_policy=context.session.extraction_policy,
            cache=analysis_record_cache_settings(context),
        )
        with analysis_records(record_request) as source:
            consumed = consume_analysis_records(
                records=source.records,
                options=build_analysis_options(context=context, corpus=corpus),
                record_sink=record_session,
            )
            _update_cache_stats(
                stats=cache_stats,
                corpus=corpus,
                source=source,
                record_count=consumed.record_count,
            )
    return RecordAnalysisResult(
        counter=consumed.counter,
        record_count=consumed.record_count,
        token_artifact=record_session.token_artifact_metadata,
    )
