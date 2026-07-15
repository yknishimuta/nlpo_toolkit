from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import closing, contextmanager

from ..analysis_records import NLPAnalysisRecord
from .errors import AnalysisCacheError
from .locking import held_cache_lock
from .models import AnalysisCacheRecordSource, AnalysisFingerprint, CacheObjectPaths
from .repository import AnalysisCacheRepository


def _is_valid(repository: AnalysisCacheRepository, paths: CacheObjectPaths) -> bool:
    if not (paths.payload.exists() and paths.metadata.exists()):
        return False
    try:
        repository.validate(paths)
    except AnalysisCacheError:
        return False
    return True


@contextmanager
def open_or_compute_analysis_records(
    *,
    repository: AnalysisCacheRepository,
    cache_key: str,
    prepared_text_sha256: str,
    prepared_text_length: int,
    fingerprint: AnalysisFingerprint,
    compute_records: Callable[[], Iterable[NLPAnalysisRecord]],
    lock_timeout_sec: float = 300.0,
) -> Iterator[AnalysisCacheRecordSource]:
    paths = repository.paths_for(cache_key)
    if _is_valid(repository, paths):
        records = repository.read(paths)
        with closing(records):  # repository.read always returns a generator
            yield AnalysisCacheRecordSource(records, "hit", cache_key, paths)
        return

    with held_cache_lock(paths.lock, timeout_sec=lock_timeout_sec):
        if _is_valid(repository, paths):
            records = repository.read(paths)
            with closing(records):
                yield AnalysisCacheRecordSource(records, "hit", cache_key, paths)
            return

        def write_through() -> Iterator[NLPAnalysisRecord]:
            with repository.writer(
                paths=paths,
                cache_key=cache_key,
                prepared_text_sha256=prepared_text_sha256,
                prepared_text_length=prepared_text_length,
                fingerprint=fingerprint,
            ) as writer:
                for record in compute_records():
                    writer.write(record)
                    yield record

        records = write_through()
        with closing(records):
            yield AnalysisCacheRecordSource(records, "miss", cache_key, paths)
