from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Mapping

from nlpo_toolkit.nlp.contracts import NLPBackendInfo

from .analysis_cache.keys import build_analysis_cache_key, prepared_text_sha256
from .analysis_cache.models import AnalysisFingerprint
from .analysis_cache.repository import AnalysisCacheRepository
from .analysis_cache.service import open_or_compute_analysis_records
from .analysis_cache.stats import AnalysisCacheRunStats
from .analysis_policy import AnalysisExtractionPolicy
from .analysis_records import (
    AnalysisOptions,
    NLPAnalysisRecord,
    evaluate_analysis_record,
    iter_nlp_analysis_records_from_text,
)
from .corpus import PreparedCorpus
from .diagnostic_trace import DiagnosticTraceWriter
from .runner_types import RunContext
from .token_artifact import (
    TokenArtifactMetadata,
    TokenArtifactWriter,
    token_artifact_metadata_path,
)

__all__ = [
    "RecordAnalysisResult",
    "RecordConsumptionResult",
    "AnalysisRecordSource",
    "build_analysis_fingerprint",
    "build_analysis_options",
    "consume_analysis_records",
    "create_analysis_cache_stats",
    "execute_record_analysis",
    "obtain_analysis_records",
]


@dataclass(frozen=True)
class RecordConsumptionResult:
    counter: Counter[str]
    record_count: int


@dataclass(frozen=True)
class RecordAnalysisResult:
    counter: Counter[str]
    record_count: int
    token_artifact: Mapping[str, object] | None
    generated_outputs: tuple[Path, ...]


@dataclass(frozen=True)
class AnalysisRecordSource:
    records: Iterator[NLPAnalysisRecord]
    cache_status: Literal["hit", "miss", "disabled"]
    cache_key: str


def _analysis_cache_dir(context: RunContext) -> Path:
    directory = Path(context.plan.config.analysis_cache.directory)
    if not directory.is_absolute():
        directory = context.plan.project_root / directory
    return directory.resolve()


def create_analysis_cache_stats(context: RunContext) -> AnalysisCacheRunStats:
    return AnalysisCacheRunStats(
        enabled=context.plan.config.analysis_cache.enabled,
        directory=str(_analysis_cache_dir(context)),
    )


def build_analysis_fingerprint(
    *,
    backend_info: NLPBackendInfo,
    policy: AnalysisExtractionPolicy,
) -> AnalysisFingerprint:
    return AnalysisFingerprint(
        backend=backend_info.name,
        language=backend_info.language,
        model=backend_info.model,
        package=backend_info.package,
        processors=policy.processors,
        chunk_size=policy.chunk_chars,
        chunk_strategy=policy.chunk_strategy,
        device=backend_info.device,
    )


def build_analysis_options(
    *,
    context: RunContext,
    corpus: PreparedCorpus,
) -> AnalysisOptions:
    filters = context.plan.config.filters
    return AnalysisOptions(
        group=corpus.label,
        source_files=tuple(corpus.files),
        use_lemma=context.plan.use_lemma,
        upos_targets=frozenset(filters.upos_targets),
        min_token_length=filters.min_token_length,
        drop_roman_numerals=filters.drop_roman_numerals,
        roman_exceptions=context.roman_exceptions,
    )


@contextmanager
def obtain_analysis_records(
    *,
    context: RunContext,
    text: str,
) -> Iterator[AnalysisRecordSource]:
    policy = context.extraction_policy
    fingerprint = build_analysis_fingerprint(
        backend_info=context.analysis_backend.info,
        policy=policy,
    )
    text_hash = prepared_text_sha256(text)
    cache_key = build_analysis_cache_key(
        prepared_text_sha256=text_hash,
        fingerprint=fingerprint,
    )
    if context.plan.config.analysis_cache.enabled:
        repository = AnalysisCacheRepository(_analysis_cache_dir(context))
        with open_or_compute_analysis_records(
            repository=repository,
            cache_key=cache_key,
            prepared_text_sha256=text_hash,
            prepared_text_length=len(text),
            fingerprint=fingerprint,
            compute_records=lambda: iter_nlp_analysis_records_from_text(
                text=text,
                nlp=context.analysis_backend.backend,
                policy=policy,
            ),
            lock_timeout_sec=context.plan.config.analysis_cache.lock_timeout_sec,
        ) as cached:
            yield AnalysisRecordSource(cached.records, cached.status, cache_key)
        return
    records = iter_nlp_analysis_records_from_text(
            text=text,
            nlp=context.analysis_backend.backend,
            policy=policy,
    )
    with closing(records):
        yield AnalysisRecordSource(records, "disabled", cache_key)


def consume_analysis_records(
    *,
    records: Iterable[NLPAnalysisRecord],
    options: AnalysisOptions,
    artifact_writer: TokenArtifactWriter | None,
    trace_writer: DiagnosticTraceWriter | None,
) -> RecordConsumptionResult:
    counter: Counter[str] = Counter()
    record_count = 0
    for raw_record in records:
        record_count += 1
        record = evaluate_analysis_record(raw_record, options=options)
        if artifact_writer is not None:
            artifact_writer.write(record)
        if trace_writer is not None:
            trace_writer.consider(record)
        if record.included and record.analysis_key:
            counter[record.analysis_key] += 1
    return RecordConsumptionResult(counter=counter, record_count=record_count)


def _token_artifact_metadata(
    *, context: RunContext, corpus: PreparedCorpus, path: Path
) -> TokenArtifactMetadata:
    plan = context.plan
    return TokenArtifactMetadata(
        group=corpus.label,
        source_files=tuple(str(file) for file in corpus.files),
        analysis_unit=plan.analysis_unit,
        upos_targets=tuple(sorted(plan.config.filters.upos_targets)),
        nlp=context.analysis_backend.info.to_dict(),
        filters={
            "min_token_length": plan.config.filters.min_token_length,
            "drop_roman_numerals": plan.config.filters.drop_roman_numerals,
        },
        artifact_path=str(path.resolve()),
    )


def _update_cache_stats(
    *,
    stats: AnalysisCacheRunStats,
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
    context: RunContext,
    corpus: PreparedCorpus,
    text: str,
    token_artifact_path: Path | None,
    trace_path: Path | None,
    cache_stats: AnalysisCacheRunStats,
) -> RecordAnalysisResult:
    artifact_writer: TokenArtifactWriter | None = None
    artifact_metadata: Mapping[str, object] | None = None
    with ExitStack() as stack:
        if token_artifact_path is not None:
            artifact_writer = stack.enter_context(
                TokenArtifactWriter(
                    token_artifact_path,
                    metadata=_token_artifact_metadata(
                        context=context, corpus=corpus, path=token_artifact_path
                    ),
                )
            )
        trace_writer: DiagnosticTraceWriter | None = None
        if trace_path is not None:
            trace_writer = stack.enter_context(
                DiagnosticTraceWriter(
                    trace_path,
                    max_rows=int(context.plan.config.trace.max_rows or 0),
                    only_keys=context.plan.config.trace.only_keys,
                    write_truncation_marker=context.plan.config.trace.write_truncation_marker,
                )
            )
        source = stack.enter_context(obtain_analysis_records(context=context, text=text))
        consumed = consume_analysis_records(
            records=source.records,
            options=build_analysis_options(context=context, corpus=corpus),
            artifact_writer=artifact_writer,
            trace_writer=trace_writer,
        )
        _update_cache_stats(
            stats=cache_stats,
            corpus=corpus,
            source=source,
            record_count=consumed.record_count,
        )

    generated: tuple[Path, ...] = ()
    if token_artifact_path is not None:
        metadata_path = token_artifact_metadata_path(token_artifact_path)
        generated = (token_artifact_path, metadata_path)
        final = artifact_writer.final_metadata if artifact_writer is not None else None
        if final is not None:
            artifact_metadata = {
                "group": corpus.label,
                "path": str(token_artifact_path.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "schema_version": final.schema_version,
                "row_count": final.row_count,
                "included_row_count": final.included_row_count,
                "complete": final.complete,
                "sha256": final.sha256,
            }
    return RecordAnalysisResult(
        counter=consumed.counter,
        record_count=consumed.record_count,
        token_artifact=artifact_metadata,
        generated_outputs=generated,
    )
