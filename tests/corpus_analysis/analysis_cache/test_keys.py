from pathlib import Path

from nlpo_toolkit.corpus_analysis.analysis_cache.keys import (
    build_analysis_cache_key, cache_object_paths, prepared_text_sha256,
)
from .conftest import fingerprint


def test_key_is_stable_and_sensitive_to_text_and_fingerprint() -> None:
    text_hash = prepared_text_sha256("Rosa")
    base = build_analysis_cache_key(
        prepared_text_sha256=text_hash, fingerprint=fingerprint()
    )
    assert base == build_analysis_cache_key(
        prepared_text_sha256=text_hash, fingerprint=fingerprint()
    )
    assert base != build_analysis_cache_key(
        prepared_text_sha256=prepared_text_sha256("rosa"), fingerprint=fingerprint()
    )
    assert base != build_analysis_cache_key(
        prepared_text_sha256=text_hash, fingerprint=fingerprint(chunk_size=101)
    )
    assert base != build_analysis_cache_key(
        prepared_text_sha256=text_hash, fingerprint=fingerprint(backend="other")
    )


def test_object_paths_use_canonical_layout(tmp_path: Path) -> None:
    key = "abcdef"
    paths = cache_object_paths(tmp_path, key)
    assert paths.payload == tmp_path / "objects/ab/abcdef.jsonl"
    assert paths.metadata == tmp_path / "objects/ab/abcdef.meta.json"
    assert paths.lock == tmp_path / "locks/ab/abcdef.lock"
