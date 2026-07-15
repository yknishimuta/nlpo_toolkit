from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_cache.codec import read_cache_metadata
from nlpo_toolkit.corpus_analysis.analysis_cache.errors import AnalysisCacheError
from nlpo_toolkit.corpus_analysis.analysis_cache.repository import AnalysisCacheRepository
from .conftest import fingerprint, record


def _writer(repository: AnalysisCacheRepository, key: str = "abcdef"):
    paths = repository.paths_for(key)
    return paths, repository.writer(
        paths=paths, cache_key=key, prepared_text_sha256="hash",
        prepared_text_length=4, fingerprint=fingerprint(),
    )


def test_writer_publishes_complete_valid_object(tmp_path: Path) -> None:
    repository = AnalysisCacheRepository(tmp_path)
    paths, writer = _writer(repository)
    with pytest.raises(AnalysisCacheError, match="not open"):
        writer.write(record())
    with writer as opened:
        opened.write(record())

    metadata = read_cache_metadata(paths.metadata)
    assert metadata.complete is True
    assert metadata.record_count == 1
    assert repository.has_candidate("abcdef")
    repository.validate(paths)
    assert list(repository.read(paths)) == [record()]


def test_writer_exception_leaves_no_metadata_or_temporary_payload(tmp_path: Path) -> None:
    repository = AnalysisCacheRepository(tmp_path)
    paths, writer = _writer(repository)
    with pytest.raises(RuntimeError):
        with writer as opened:
            opened.write(record())
            raise RuntimeError("stop")
    assert not paths.metadata.exists()
    assert not paths.payload.exists()
    assert not list(paths.payload.parent.glob("*.tmp"))


def test_repository_requires_both_payload_and_metadata(tmp_path: Path) -> None:
    repository = AnalysisCacheRepository(tmp_path)
    paths = repository.paths_for("abcdef")
    paths.payload.parent.mkdir(parents=True)
    paths.payload.touch()
    assert repository.has_candidate("abcdef") is False
    paths.payload.unlink()
    paths.metadata.touch()
    assert repository.has_candidate("abcdef") is False
