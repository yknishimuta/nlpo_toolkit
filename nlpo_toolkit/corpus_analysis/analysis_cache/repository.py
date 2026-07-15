from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..analysis_records import NLPAnalysisRecord
from .codec import read_analysis_records, validate_cache_object
from .keys import cache_object_paths
from .models import AnalysisCacheMetadata, AnalysisFingerprint, CacheObjectPaths
from .writer import AnalysisCacheWriter


class AnalysisCacheRepository:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir).resolve()

    def paths_for(self, cache_key: str) -> CacheObjectPaths:
        return cache_object_paths(self.cache_dir, cache_key)

    def has_candidate(self, cache_key: str) -> bool:
        paths = self.paths_for(cache_key)
        return paths.payload.exists() and paths.metadata.exists()

    def validate(self, paths: CacheObjectPaths) -> AnalysisCacheMetadata:
        return validate_cache_object(paths)

    def read(self, paths: CacheObjectPaths) -> Iterator[NLPAnalysisRecord]:
        return read_analysis_records(paths)

    def writer(
        self,
        *,
        paths: CacheObjectPaths,
        cache_key: str,
        prepared_text_sha256: str,
        prepared_text_length: int,
        fingerprint: AnalysisFingerprint,
    ) -> AnalysisCacheWriter:
        return AnalysisCacheWriter(
            paths=paths,
            cache_key=cache_key,
            prepared_text_sha256=prepared_text_sha256,
            prepared_text_length=prepared_text_length,
            fingerprint=fingerprint,
        )
