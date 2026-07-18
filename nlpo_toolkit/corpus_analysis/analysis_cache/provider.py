from __future__ import annotations

from contextlib import closing, contextmanager
from collections.abc import Iterator

from nlpo_toolkit.nlp.contracts import NLPBackendInfo

from ..analysis_policy import AnalysisExtractionPolicy
from ..analysis_records import iter_nlp_analysis_records_from_text
from ..ports import AnalysisRecordRequest, AnalysisRecordSource
from .keys import build_analysis_cache_key, prepared_text_sha256
from .models import AnalysisFingerprint
from .repository import AnalysisCacheRepository
from .service import open_or_compute_analysis_records


def build_analysis_fingerprint(
    *, backend_info: NLPBackendInfo, policy: AnalysisExtractionPolicy
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


@contextmanager
def provide_analysis_records(
    request: AnalysisRecordRequest,
) -> Iterator[AnalysisRecordSource]:
    fingerprint = build_analysis_fingerprint(
        backend_info=request.backend.info,
        policy=request.extraction_policy,
    )
    text_hash = prepared_text_sha256(request.text)
    cache_key = build_analysis_cache_key(
        prepared_text_sha256=text_hash,
        fingerprint=fingerprint,
    )

    def compute_records():
        return iter_nlp_analysis_records_from_text(
            text=request.text,
            nlp=request.backend.backend,
            policy=request.extraction_policy,
        )

    if request.cache.enabled:
        repository = AnalysisCacheRepository(request.cache.directory)
        with open_or_compute_analysis_records(
            repository=repository,
            cache_key=cache_key,
            prepared_text_sha256=text_hash,
            prepared_text_length=len(request.text),
            fingerprint=fingerprint,
            compute_records=compute_records,
            lock_timeout_sec=request.cache.lock_timeout_sec,
        ) as cached:
            yield AnalysisRecordSource(
                records=cached.records,
                cache_status=cached.status,
                cache_key=cache_key,
            )
        return

    records = compute_records()
    with closing(records):
        yield AnalysisRecordSource(
            records=records,
            cache_status="disabled",
            cache_key=cache_key,
        )
