from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.corpus_analysis.analysis_cache.keys import (
    build_analysis_cache_key, prepared_text_sha256,
)
from nlpo_toolkit.corpus_analysis.analysis_cache.repository import AnalysisCacheRepository
from nlpo_toolkit.corpus_analysis.analysis_cache.service import open_or_compute_analysis_records
from .conftest import fingerprint, record


def _open(repository: AnalysisCacheRepository, compute):
    fp = fingerprint()
    text_hash = prepared_text_sha256("Rosa")
    key = build_analysis_cache_key(prepared_text_sha256=text_hash, fingerprint=fp)
    return key, open_or_compute_analysis_records(
        repository=repository, cache_key=key,
        prepared_text_sha256=text_hash, prepared_text_length=4,
        fingerprint=fp, compute_records=compute,
    )


def test_miss_then_hit_round_trip_and_lock_cleanup(tmp_path: Path) -> None:
    repository = AnalysisCacheRepository(tmp_path)
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return iter((record(),))

    key, opened = _open(repository, compute)
    with opened as source:
        assert source.status == "miss"
        assert list(source.records) == [record()]
    paths = repository.paths_for(key)
    assert paths.payload.exists() and paths.metadata.exists()
    assert not paths.lock.exists()

    _, opened = _open(repository, lambda: pytest.fail("compute called on hit"))
    with opened as source:
        assert source.status == "hit"
        assert list(source.records) == [record()]
    assert calls == 1
    assert not paths.lock.exists()


def test_early_exit_does_not_commit_and_releases_lock(tmp_path: Path) -> None:
    repository = AnalysisCacheRepository(tmp_path)
    key, opened = _open(repository, lambda: iter((record(), record("amat"))))
    with opened as source:
        assert next(source.records) == record()
    paths = repository.paths_for(key)
    assert not paths.metadata.exists()
    assert not paths.lock.exists()
    assert not list(paths.payload.parent.glob("*.tmp"))


def test_compute_failure_propagates_without_complete_object(tmp_path: Path) -> None:
    repository = AnalysisCacheRepository(tmp_path)

    def compute():
        yield record()
        raise RuntimeError("boom")

    key, opened = _open(repository, compute)
    with pytest.raises(RuntimeError, match="boom"):
        with opened as source:
            list(source.records)
    paths = repository.paths_for(key)
    assert not paths.metadata.exists()
    assert not paths.lock.exists()


def test_corrupt_payload_is_recomputed(tmp_path: Path) -> None:
    repository = AnalysisCacheRepository(tmp_path)
    key, opened = _open(repository, lambda: iter((record(),)))
    with opened as source:
        list(source.records)
    paths = repository.paths_for(key)
    paths.payload.write_text("corrupt\n", encoding="utf-8")
    with _open(repository, lambda: iter((record("amat"),)))[1] as source:
        assert source.status == "miss"
        assert list(source.records) == [record("amat")]
